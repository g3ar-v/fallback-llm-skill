import openai as ai
import os


from threading import Thread
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from core import Skill, intent_file_handler
from core.skills.common_query_skill import CommonQuerySkill, CQSMatchLevel

model = "gpt-3.5-turbo"

# TODO: BUILD BETTER PROMPT
template = """ You're not just a personal assistant; you're a knowledgeable and witty
companion. With a vast range of literary references, including the works of Alan Grants,
you can provide clever and insightful responses. I address you as "Sir" in a formal
manner, maintaining a respectful tone. While occasional moments of casualness may arise,
our overall interaction will predominantly be in a formal tone. If you encounter a
request outside your areas of expertise, you will provide a disclosure on your limited
knowledge in that field. For example, when unsure about an answer, you may respond with
"I don't know" or a similar acknowledgement. Furthermore, if you lack the necessary
capabilities to perform a specific action, I will inform you by saying "I don't have the
capabilities for that." In our conversations, feel free to explore various topics, seek
answers to my queries, or simply engage in interesting dialogue. you're here to provide
nsightful responses and exchange thoughts. Any query that involve "create" and you have
no idea how to do this just say you don't have the capabilities
{history}
Human: {input}
Assistant:"""

prompt = PromptTemplate(input_variables=["history", "input"], template=template)


class FallbackGPT(CommonQuerySkill, Skill):
    def __init__(self):
        super().__init__()
        self.key = self.settings.get("key")

    def initialize(self):
        os.environ["OPENAI_API_KEY"] = self.key
        self.llm = OpenAI(temperature=0)
        self.memory = ConversationBufferMemory()

    def CQS_match_query_phrase(self, utt):
        response = self.handle_fallback_GPT(utt)
        if response:
            return (utt, CQSMatchLevel.CATEGORY, response)
        return None

    def handle_fallback_GPT(self, message):
        """Handles qa utterances """

        self.memory.chat_memory.add_user_message(message)

        try:
            # completion = self.chatgpt.create(model=model, messages=prompt,
            #                                  max_tokens=100, temperature=0.2)
            gptchain = LLMChain(llm=self.llm, verbose=True, memory=self.memory,
                                prompt=prompt)
            # response = completion.choices[0].message["content"]
            response = gptchain.predict(input=message)
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
                                         memory=ConversationBufferWindowMemory(k=2),
                                         prompt=prompt)
        return conversation.predict(input=query)

    def converse(self, message=None):
        utterance = message.data.get('utterances', [""])[-1]
        # Not dynamic enough, maybe a confidence threshold later depends on if I want
        # to stop the conversation
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

    def chatgpt(self):
        if not self.key:
            raise ValueError("Openai key not set in settings.json")
        ai.api_key = self.key
        return ai.ChatCompletion

    def stop(self):
        pass


def create_skill():
    return FallbackGPT()
