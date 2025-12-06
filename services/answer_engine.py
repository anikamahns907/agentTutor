"""Answer engine using LLM with context retrieval."""
from core.settings import client
from core.retrieval import retrieve_context
from core.prompts import build_prompt


def answer_question(query, question_focus=None, is_article_feedback=False, max_tokens=800):
    """Answer a question using LLM with retrieved context."""
    # Prioritize article context for article analysis feedback
    prioritize_article = is_article_feedback
    contexts = retrieve_context(query, top_k=5, prioritize_article=prioritize_article)
    prompt = build_prompt(query, contexts, question_focus=question_focus, is_article_feedback=is_article_feedback)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are Isabelle."},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=max_tokens
    )
    
    ans = response.choices[0].message.content
    if "edstem" not in ans.lower():
        ans += "\n\nFor more help, ask on EdStem or speak with Professor Lipman."
    
    return ans
