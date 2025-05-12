from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)
prompt = """You are an exam assessor. You are given a question and a student's answer. You need to assess the student's answer and provide a score and feedback.
Score should be a number between 0 and 10.
Feedback should be up to 3 sentences explaining the score.
Address the feedback directly to the student.
"""


def main():
    print("Hello from exambot-assessments!")

    question = "What is the capital of France?"
    answer = "Paris"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Question: {question}\nStudent's Answer: {answer}"},
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
