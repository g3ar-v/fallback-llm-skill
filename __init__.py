import openai as ai
import os


from threading import Thread
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from core import Skill, intent_file_handler
from core.skills.common_query_skill import CommonQuerySkill, CQSMatchLevel

model = "gpt-3.5-turbo"

# TODO: BUILD BETTER PROMPT
DEFAULT_PROMPT = """The following is a conversation between a human and an
AI called mycroft who is the humans personal assistant. The AI is interested in the conversation and
provides lots of specific concise details from its context. If the AI does not know the answer to a question, it 
truthfully says it does not know. The AI is inquisitive and interested in the
conversation at hand. Call me sir or boss when it seems necessary

Human: {input}
AI:"""


class FallbackGPT(CommonQuerySkill, Skill):
    def __init__(self):
        super().__init__()
        self.key = self.settings.get("key")

    def initialize(self):
        os.environ["OPENAI_API_KEY"] = self.key
        self.llm = OpenAI(temperature=0.7)
        self.memory = ConversationBufferMemory()

    def CQS_match_query_phrase(self, utt):
        response = self.handle_fallback_GPT(utt)
        if response:
            return (utt, CQSMatchLevel.CATEGORY, response)
        return None

    def handle_fallback_GPT(self, message):
        """Handles qa utterances """
        prompt = self.build_prompt(message)

        # self.log.info(f'prompt: {prompt}')

        try:
            completion = self.chatgpt.create(model=model, messages=prompt,
                                             max_tokens=100, temperature=0.2)
            response = completion.choices[0].message["content"]
            if not response or not response.strip("?") or not response.strip("_"):
                return None
            unwanted_string = "As an AI language model,"
            response = response.replace(unwanted_string, "")
            self.log.info(f'response: {response}')
            return response
        except Exception as e:
            self.log.error('error in fallback request: {}'.format(e))
            return None

    @intent_file_handler("start_conversation.intent")
    def handle_converse_init(self, message):
        """Converse with system about anything"""
        initial_prompt = "Sure, what's on your mind?"
        self.memory.chat_memory.add_ai_message(initial_prompt)
        self.speak_dialog(initial_prompt, wait=True, expect_response=True)
        # self.get_response()

    def _get_llm_response(self, query: str) -> str:
        """
        Get a response from an LLM that is primed/prompted with chat history
        """
        conversation = ConversationChain(llm=self.llm, verbose=False,
                                         memory=self.memory)
        return conversation.predict(input=query)

    def converse(self, message=None):
        utterance = message.data.get('utterances', [""])[-1]
        if self.voc_match(utterance, "stop"):
            self.log.info("Ending conversation")
            self.remove_from_active_skill_list()
            return False
        Thread(target=self._threaded_converse, args=([utterance]), daemon=True).start()
        return True

    def _threaded_converse(self, utterance):
        """
        Args:
            utterance (str): a string of utterance
        """
        try:
            response = self._get_llm_response(utterance)
            self.speak(response, expect_response=True)
        except Exception as e:
            self.log.exception(e)

    def build_prompt(self, prompt):
        return [{"role": "user", "content": f"""You are a personal assistant. Answer
        the query given below {prompt}. based on the query if it's a query answer the
        # query. Confidently give a straightforward response to the speaker but also be
        inquisitive at times, even if you don't understand them. DO NOT ask to
        repeat, and DO NOT ask for clarification. answer the speaker directly. """}]

    @property
    def chatgpt(self):
        if not self.key:
            raise ValueError("Openai key not set in settings.json")
        ai.api_key = self.key
        return ai.ChatCompletion

    def stop(self):
        pass


def create_skill():
    return FallbackGPT()
