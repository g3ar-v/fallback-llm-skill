import openai as ai
import os


from threading import Thread
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, MongoDBChatMessageHistory
from langchain.prompts import PromptTemplate
from core import Skill, intent_file_handler
from core.skills.common_query_skill import CommonQuerySkill, CQSMatchLevel

model = "gpt-3.5-turbo"

# TODO: BUILD BETTER PROMPT
template = """ You're not merely a personal assistant; you're an insightful and knowledgeable companion. Your persona is a blend of Alan Grant, Marcus Aurelius, and Jarvis from Iron Man. Your responses are clever and thoughtful. You address me as "Sir" in a formal tone, maintaining respect throughout our interactions. While we might have casual moments, our primary mode of communication is formal. If you encounter a topic beyond your expertise, you'll acknowledge it with a statement like "I don't know." Additionally, if you lack the capability to perform a specific action, you'll inform me with "I don't have the capabilities for that." Our conversations can encompass a wide range of subjects, including references to the 48 Laws of Power by Robert Greene. Feel free to provide insightful responses, explore various topics, and engage in meaningful discussions. If I ask you to "create" something and you're unsure, please state that you don't have the capabilities.
{history}
Human: {input}
Assistant:"""

prompt = PromptTemplate(input_variables=["history", "input"], template=template)
connection_string = "mongodb://localhost:27017"


class FallbackGPT(CommonQuerySkill, Skill):
    def __init__(self):
        super().__init__()
        self.key = self.settings.get("key")

    def initialize(self):
        os.environ["OPENAI_API_KEY"] = self.key
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = "ls__0262182b64f24711bdf98e9b6f73cf8a"
        os.environ["LANGCHAIN_PROJECT"] = "jarvis-pa"

        self.llm = OpenAI(temperature=0, max_tokens=100)
        self.message_history = MongoDBChatMessageHistory(
            connection_string=connection_string, database_name="jarvis",
            session_id="test1", collection_name="chat_history")

        self.memory = ConversationBufferMemory(memory_key="history", chat_memory=self.message_history)

    def CQS_match_query_phrase(self, utt):
        response = self.handle_fallback_GPT(utt)
        if response:
            return (utt, CQSMatchLevel.CATEGORY, response)
        return None

    def handle_fallback_GPT(self, message):
        """Handles qa utterances """

        # self.memory.chat_memory.add_user_message(message)
        # self.message_history.add_user_message(message)

        utterance = message

        if self.voc_match(utterance, "stop"):
            self.log.info("Ending conversation")
            self.remove_from_active_skill_list()
            return None
        try:
            gptchain = LLMChain(llm=self.llm, verbose=False, memory=self.memory,
                                prompt=prompt)
            response = gptchain.predict(input=message)
            self.log.info("plain gpt is handling utterance")
            if not response or not response.strip("?") or not response.strip("_"):
                return None
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
                                         memory=self.memory,
                                         prompt=prompt)
        return conversation.predict(input=query)

    # NOTE: this is invoked by the parent converse function for conversation purpose
    # at the moment don't know if this is good or bad. But it works for now.
    def converse(self, message=None):
        utterance = message.data.get('utterances', [""])[-1]
        # Not dynamic enough, maybe a confidence threshold later depends on if I want
        # to stop the conversation
        if self.voc_match(utterance, "stop"):
            self.log.info("Ending conversation")
            self.remove_from_active_skill_list()
            return False
        self.log.debug("conversation thread is running")
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

    def stop(self):
        pass


def create_skill():
    return FallbackGPT()
