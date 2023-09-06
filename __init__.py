import os
import subprocess

from langchain import LLMChain
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.llms.openai import OpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import VectorStoreRetrieverMemory, ConversationBufferMemory, MongoDBChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings

from core.intent_services import AdaptIntent
from core import Skill, intent_file_handler, intent_handler
from core.skills.common_query_skill import CommonQuerySkill, CQSMatchLevel
from pymongo import MongoClient

model = "gpt-3.5-turbo"

# TODO: BUILD BETTER PROMPT
template = """You're a personal assistant; you're a witty, insightful and knowledgeable companion with a high sense of humour. Your persona is a blend of Alan Watts, JARVIS from Iron Man meaning. Your responses are clever and thoughtful with brevity. Often you provide responses in a style reminiscent of Alan Watts. You address me as "Sir" in a formal tone, throughout our interactions. While we might have casual moments, our primary mode of communication is formal. Occasionally you relate other fields with the field we are talking about. You understand slangs used in everyday dialogs. Like "running rings" means "to be smarter or better than someone at something". If there's an opportunity for banter and tease given the context use it.
relevant pieces of previous conversation:
{history}
(You do not need to use these pieces of information if not relevant)
Current conversation:
Human: {input}
Assistant:"""

# {history}

# NOTE: removed history cause of token cost
# add history to input variable list
prompt = PromptTemplate(input_variables=["history", "input"], template=template)
# NOTE: for security reasons you might want to store connection string as env var
as_prompt = SystemMessage(content="""You are an expert at using applescript commands. Only provide an excecutable
                               line or block of applescript code as output. Never output any text before or after the
                               code, as the output will be directly exectued in a shell. Key details to take note of: 
                               I use google chrome as my browser""")

embeddings = OpenAIEmbeddings()


class FallbackGPT(CommonQuerySkill, Skill):
    def __init__(self):
        super().__init__()
        self.openai_key = self.config_core["microservices"].get("openai_key")
        self.langsmith_key = self.config_core["microservices"].get("langsmith_key")
        self.conn_string = self.config_core["microservices"].get("mongo_conn_string")

    def initialize(self):
        os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") or self.openai_key
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = self.langsmith_key
        os.environ["LANGCHAIN_PROJECT"] = "jarvis-pa"
        client = MongoClient(self.conn_string)
        db_name = "jarvis"
        collection_name = "chat_history"
        self.collection = client[db_name][collection_name]

        self.llm = OpenAI(temperature=0.7, max_tokens=100)
        self.message_history = MongoDBChatMessageHistory(
            connection_string=self.conn_string, database_name="jarvis",
            session_id="main", collection_name="chat_history")

        self.vectorstore = MongoDBAtlasVectorSearch(self.collection, embeddings, index_name="default")
        # self.memory = ConversationBufferMemory(memory_key="history",
        #                                        chat_memory=self.message_history)

        retriever = self.vectorstore.as_retriever(search_kwargs=dict(k=1))
        self.memory = VectorStoreRetrieverMemory(retriever=retriever)

    def CQS_match_query_phrase(self, utt):
        response = self.handle_fallback_GPT(utt)
        if response:
            return (utt, CQSMatchLevel.CATEGORY, response)
        return None

    # NOTE: with a chat history and index there'd be no need for a converse method because 
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
            conversation = ConversationChain(llm=self.llm,
                                             verbose=True, memory=self.memory, prompt=prompt)
            response = conversation.predict(input=message)
            self.log.info("LLM is handling utterance")
            return response
        except Exception as e:
            self.log.error('error in fallback request: {}'.format(e))
            return None

    @intent_handler(AdaptIntent().require("query"))
    def handle_mac_script_exec(self, message):
        prompt = message.data.get('utterances', [""])[-1]
        chat = ChatOpenAI(model_name=model, openai_api_key=self.openai_key)

        response = chat.generate(
            messages=[
                [
                    as_prompt,
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
    return FallbackGPT()


if __name__ == "__main__":
    # Your code here
    gpt = FallbackGPT()
    gpt.initialize()
    gpt.retrieval("what was I saying earlier")
