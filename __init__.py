import openai as ai
import os


from threading import Thread
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from core import Skill, intent_file_handler, Message
from core.skills.common_query_skill import CommonQuerySkill, CQSMatchLevel

model = "gpt-3.5-turbo"

DEFAULT_PROMPT = """The following is a friendly conversation between a human and an
AI. The AI is talkative and provides lots of specific details from its context. If the
AI does not know the answer to a question, it truthfully says it does not know.

Human: {input}
AI:"""


class FallbackGPT(CommonQuerySkill, Skill):
    def __init__(self):
        super().__init__()
        self.key = self.settings.get("key")

    def initialize(self):
        self.llm = OpenAI(temperature=0.2)
        self.memory = ConversationBufferMemory()
        os.environ["OPENAI_API_KEY"] = self.key

    def CQS_match_query_phrase(self, utt):
        response = self.handle_fallback_GPT(utt)
        if response:
            return (utt, CQSMatchLevel.CATEGORY, response)
        return None

    @property
    def chat_timeout(self):
        """The chat_timeout in seconds."""
        return self.settings.get("chat_timeout") or 300

    def handle_fallback_GPT(self, message):
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
        initial_prompt = "What do you want to talk about?"
        self.memory.chat_memory.add_ai_message(initial_prompt)
        self.speak_dialog(initial_prompt, wait=True)
        self.make_active()
        self.bus.emit(Message("core.mic.listen"))

    def _get_llm_response(self, query: str) -> str:
        """
        Get a response from an LLM that is primed/prompted with chat history
        """
        conversation = ConversationChain(llm=self.llm, verbose=True,
                                         memory=self.memory)
        return conversation.predict(input=query)

    def converse(self, message=None):
        # Track timeout and return false.
        # if time() -
        utterance = message.data.get('utterances', [""])[-1]
        if self.voc_match(utterance, "stop"):
            self.speak_dialog("ending conversation")
            return False
        self.log.info("utterance: {}".format(utterance))
        Thread(target=self._threaded_converse, args=([utterance]), daemon=True).start()
        return True

    def _threaded_converse(self, utterance):
        try:
            response = self._get_llm_response(utterance)
            self.speak(response, expect_response=True if response.endswith("?") or "?"
                       in response else False)
        except Exception as e:
            self.log.exception(e)

    def build_prompt(self, prompt):
        return [{"role": "user", "content": f"""You are a personal assistant. Answer
        the query given below {prompt}. based on the query if it's a query answer the
        query. Confidently give a straightforward response to the speaker but also be
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
