import subprocess

from langchain.chains import LLMChain
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import VectorStoreRetrieverMemory, ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.embeddings.openai import OpenAIEmbeddings

from core.intent_services import AdaptIntent
from core import Skill, intent_handler
from core.skills.common_query_skill import CommonQuerySkill, CQSMatchLevel
from core.llm import LLM, main_persona_prompt, prompt_to_osa
from core.util.time import now_local
from pymongo import MongoClient

model = "gpt-3.5-turbo"

applescript_prompt = SystemMessage(content=prompt_to_osa)

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
        retriever = self.vectorstore.as_retriever(search_kwargs=dict(k=1))
        self.relevant_memory = VectorStoreRetrieverMemory(retriever=retriever)

    def CQS_match_query_phrase(self, utt):
        response = self.handle_fallback_GPT(utt)
        if response:
            return (utt, CQSMatchLevel.CATEGORY, response)
        return None

    def handle_fallback_llm(self, message):
        """Handles qa utterances"""

        today = now_local()
        date_str = today.strftime("%B %d, %Y")
        time_str = today.strftime("%I:%M %p")

        try:
            rel_mem = self.relevant_memory.load_memory_variables({"prompt": message})[
                "history"
            ]
            chat_history = self.chat_history.load_memory_variables({})["chat_history"]
            llm = LLMChain(llm=self.llm.model, verbose=True, prompt=main_persona_prompt)
            response = llm.predict(
                input=message,
                date_str=date_str + ", " + time_str,
                rel_mem=rel_mem,
                curr_conv=chat_history,
            )
            self.log.info("LLM is handling utterance")
            return response
        except Exception as e:
            self.log.error("error in fallback request: {}".format(e))
            return None

    @intent_handler(AdaptIntent().require("query"))
    def handle_mac_script_exec(self, message):
        prompt = message.data.get("utterances", [""])[-1]
        chat = ChatOpenAI(model_name=model, openai_api_key=self.openai_key)

        response = chat.generate(
            messages=[
                [
                    applescript_prompt,
                    HumanMessage(content=f"Here's what I'm trying to do: {prompt}"),
                ]
            ]
        )
        script = response.generations[0][0].message.content
        # Execute the script
        self.log.info(script)
        try:
            self.speak_dialog("confirmation")
            output = subprocess.check_output(["osascript", "-e", script])
            # self.speak(output.decode())
            self.log.debug(output.decode())
        except subprocess.CalledProcessError as e:
            self.speak_dialog("I couldn't carry out that operation, Sir")
            self.log.error(f"Script execution failed: {e}")
        except OSError as e:
            self.speak_dialog("no_osascript")
            self.log.error(f"osascript not found: {e}")

    def stop(self):
        pass


def create_skill():
    return FallbackLLM()
