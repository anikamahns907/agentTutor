"""Question generation functions."""
def generate_assignment_questions(article_title: str):
    """Generate the standard 10 questions for article analysis."""
    return [
        {
            "question": "What statistical methods are used in this study?",
            "focus": "Identifying methods",
            "hint": "Look for mentions of tests, confidence intervals, p-values, regression, etc."
        },
        {
            "question": "What is the study design? (e.g., randomized controlled trial, observational study, etc.)",
            "focus": "Study design",
            "hint": "Consider how participants were selected and assigned to groups"
        },
        {
            "question": "How are the results interpreted? What do the statistical findings tell us?",
            "focus": "Interpretation",
            "hint": "Look at confidence intervals, p-values, and what conclusions are drawn"
        },
        {
            "question": "What are the limitations of the statistical analysis?",
            "focus": "Critical thinking",
            "hint": "Consider sample size, assumptions, potential biases, etc."
        },
        {
            "question": "How do the statistical methods used relate to concepts from our course?",
            "focus": "Course connection",
            "hint": "Connect to lecture materials, textbook concepts, and class discussions"
        },
        {
            "question": "How would you explain the methods used to colleagues that don't have any statistics background?",
            "focus": "Communication",
            "hint": "Think about how to translate technical statistical concepts into plain language"
        },
        {
            "question": "What's a 1-2 sentence summary of the main findings? Be as clear/concise as possible while still maintaining technical accuracy.",
            "focus": "Summary",
            "hint": "Focus on the key statistical findings and their practical significance"
        },
        {
            "question": "If you had this data, would there be another analysis you'd perform to gain more insights? Explain it without technical jargon.",
            "focus": "Critical thinking",
            "hint": "Consider what additional questions could be answered or what alternative approaches might be valuable"
        },
        {
            "question": "Would you recommend the authors communicate their findings differently? What changes would improve clarity?",
            "focus": "Communication",
            "hint": "Think about how statistical results are presented and whether they could be more accessible"
        },
        {
            "question": "Pick a specific piece of output (like a p-value or summary statistic). Interpret it.",
            "focus": "Interpretation",
            "hint": "Choose a specific statistical result from the article and explain what it means in practical terms"
        }
    ]
