import openai as ai

from core import Skill
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
        pass

    def CQS_match_query_phrase(self, utt):
        response = self.handle_fallback_GPT(utt)
        if response:
            return (utt, CQSMatchLevel.CATEGORY, response)
        return None

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

    # def converse()

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
