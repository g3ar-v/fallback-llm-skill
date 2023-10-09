import interpreter

from langchain.chains import LLMChain
from langchain.memory import VectorStoreRetrieverMemory, ConversationBufferWindowMemory
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.embeddings.openai import OpenAIEmbeddings

from core.intent_services import AdaptIntent
from core import Skill, intent_handler
from core.skills.common_query_skill import CommonQuerySkill, CQSMatchLevel
from core.llm import LLM, main_persona_prompt
from core.util.time import now_local
from pymongo import MongoClient

embeddings = OpenAIEmbeddings()


class FallbackLLM(CommonQuerySkill, Skill):
    def __init__(self):
        super().__init__()

    def initialize(self):
        self.llm = LLM()
        from core.configuration.config import Configuration

        config = Configuration.get()
        self.openai_key = config["microservices"]["openai_key"]

        client = MongoClient(LLM.conn_string)
        db_name = "jarvis"
        collection_name = "chat_history"
        self.collection = client[db_name][collection_name]

        # NOTE: get chat history lines
        self.message_history = self.llm.message_history
        self.chat_history = ConversationBufferWindowMemory(
            memory_key="chat_history",
            chat_memory=self.message_history,
            ai_prefix="Jarvis",
            k=3,
        )

        # NOTE: get relevant information from chat history
        self.vectorstore = MongoDBAtlasVectorSearch(
            self.collection, embeddings, index_name="default"
        )
        retriever = self.vectorstore.as_retriever(
            search_kwargs=dict(k=1), search_type="mmr"
        )
        self.relevant_memory = VectorStoreRetrieverMemory(retriever=retriever)

    def CQS_match_query_phrase(self, utt):
        response = self.handle_fallback_llm(utt)
        if response:
            return (utt, CQSMatchLevel.CATEGORY, response)
        return None

    def handle_fallback_llm(self, message):
        """Handles qa utterances"""

        today = now_local()
        date_str = today.strftime("%B %d, %Y")
        time_str = today.strftime("%I:%M %p")

        try:
            # rel_mem = self.relevant_memory.load_memory_variables({"prompt": message})[
            #     "history"
            # ]

            chat_history = self.chat_history.load_memory_variables({})["chat_history"]
            llm = LLMChain(llm=self.llm.model, verbose=True, prompt=main_persona_prompt)
            response = llm.predict(
                input=message,
                date_str=date_str + ", " + time_str,
                rel_mem=None,
                curr_conv=chat_history,
            )
            self.log.info("LLM is handling utterance")
            return response
        except Exception as e:
            self.log.error("error in fallback request: {}".format(e))
            return None

    @intent_handler(AdaptIntent().require("Query"))
    def handle_mac_script_exec(self, message):
        utterance = message.data.get("utterances", [""])[-1]
        interpreter.model = "gpt-3.5-turbo"
        interpreter.auto_run = True
        interpreter.chat(utterance, display=False)
        last_message = interpreter.messages[-1]["message"]
        self.log.info("length of message: " + str(len(last_message)))
        if len(last_message) <= 225:
            # check for new line to get the last sentence
            if "\n" in last_message:
                last_message = last_message.split("\n")[:-1]
            self.log.info(last_message)
            self.speak(last_message)
        else:
            self.speak_dialog("Confirmation")

    def stop(self):
        pass


def create_skill():
    return FallbackLLM()


if __name__ == "__main__":
    from core.messagebus import Message

    skill = FallbackLLM()
    skill.initialize()
    message = Message(
        "adapt", {"utterances": ["play music from spotify application on my mac"]}
    )
    skill.handle_mac_script_exec(message)
