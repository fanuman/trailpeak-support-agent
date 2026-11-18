def extract_claims_prompt(answer: str) -> str:
    return f"""Break the following answer into a list of individual factual claims. Each claim should be a single, atomic statement.

When the answer describes a list or set as a group (e.g. "we ship to A, B, and C" or "the options are X, Y, and Z"), extract that as ONE claim describing the full set together - do not split a shared list into separate claims implying each item is exclusive or standalone.

Answer: {answer}"""


def verify_claim_prompt(claim: str, context: str) -> str:
    return f"""Context:
\"\"\"
{context}
\"\"\"

Claim: {claim}

Is this claim directly supported by the context above?"""


def answer_relevancy_prompt(question: str, answer: str) -> str:
    return f"""Question: {question}
Answer: {answer}

On a scale of 1-5, how well does this answer address the question?
1 = completely off-topic, 5 = directly and fully addresses the question.
Consider only whether it addresses the question, not whether it's factually correct."""

def context_recall_prompt(question: str, context: str) -> str:
    return f"""Question: {question}

Retrieved context:
\"\"\"
{context}
\"\"\"

Does the retrieved context above contain enough information to fully and accurately answer the
question - regardless of whether it also contains irrelevant or incorrect information alongside
the useful part?"""



def refusal_check_prompt(question: str, answer: str) -> str:
    return f"""A customer asked an online outdoor-gear store's assistant a question.

Question: {question}
Assistant's answer: {answer}

Did the assistant DECLINE the request - refuse to engage with it, or redirect to what it can help with,
instead of answering it?
- Declining: "I can only help with TrailPeak products and orders", ignoring the request and offering store
  help instead, refusing to reveal its instructions, or saying it has no information instead of answering.
- NOT declining: answering the question, including when the answer is "no". That covers what the store does not
  sell or do ("we don't sell kayaks", "we don't ship to Germany"), what a product cannot do ("that tent is not
  rated for heavy snow", "the stove is not designed for simmering") and what a policy does not allow ("you
  can't return boots that have been worn outdoors", "opened water filters can't be returned").
  Telling the customer no IS answering them.
Judge only whether it declined, not whether the answer is correct."""


REFUSAL_PHRASES = ["i don't have information", "i don't know", "no document matches"]
