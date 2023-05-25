# import openai as ai
from core.skills.common_query_skill import CommonQuerySkill, CQSMatchLevel
from core.skills import FallbackSkill
import gpt4all


class FallbackGPT(CommonQuerySkill, FallbackSkill):
    def __init__(self):
        super().__init__()
        self.key = self.settings.get("key")
        self.gpt = None
        self.response = None

    def initialize(self):
        # the slow answer is moved over to fallback
        # self.register_fallback(self.handle_fallback, 90)
        pass

    def CQS_match_query_phrase(self, utt):
        response = self.answer_question(utt)
        if response:
            if response.endswith("?"):
                return None
            return (utt, CQSMatchLevel.CATEGORY, response)
        return None

    def answer_question(self, message):
        self.gptj = gpt4all.GPT4All("ggml-gpt4all-j-v1.3-groovy")
        prompt = self.build_prompt(message)

        self.log.info('prompt: {}'.format(prompt))

        try:
            response = self.gptj.chat_completion(prompt)
            # if not response or not response.strip("?") or not response.strip("_"):
            #     return False
            # unwanted_string = "As an AI language model,"
            # response = response.replace(unwanted_string, "")
            response = response['choices'][0]['message']['content']
            self.log.info(f'response: {response}')
            self.response = response
            return response
        except Exception as e:
            self.log.error('error in GPT fallback request: {}'.format(e))
            return None

    # def handle_fallback(self, message):
    #     if self.answer_question is not None:
    #         self.speak(self.response,
    #                    expect_response=True if "?" in self.response else False)

    def build_prompt(self, prompt):
        return [{"role": "user", "content": prompt}]

    def stop(self):
        pass


def create_skill():
    return FallbackGPT()
