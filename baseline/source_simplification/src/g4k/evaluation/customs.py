"""A module that defines custom examples and prompts."""

from langchain_core.pydantic_v1 import BaseModel
from ragas.llms.output_parser import get_json_format_instructions
from ragas.llms.prompt import Prompt


class QuestionFilter(BaseModel):
    """A class necessary to create a Prompt object."""

    feedback: str
    verdict: int


custom_examples = [
    {
        "context": """{'Kennzahlen': 'Jahresueberschuss',
                '2012': 427.0, '2013': 272.0, '2014': 535.0, '2015': 634.0,
                '2016': 743.0, '2017': 790.0, '2018': 1075.0, '2019': 1080.0,
                '2020': 368.0, '2021': 1169.0, '2022': 2179.0, '2023': 3137.0,
                'Unternehmen': 'Bayer'}""",
        "question": "Which year had the highest yearly surplus for Bayer??",
        "answer": {
            "answer": "The year with the highest yearly surplus for Bayer is 2023.",
            "verdict": 1,
        },
    },
    {
        "context": """{'Kennzahlen': 'Cashflow',
                '2012': 707.7, '2013': 728.3, '2014': 677.3, '2015': 10.1,
                '2016': 1621.4, '2017': 1056.2, '2018': 1298.2, '2019': 926.1,
                '2020': 1412.0, '2021': 908.9, '2022': 2483.6, '2023': 2549.0,
                'Unternehmen': 'BASF'}""",
        "question": "How did BASF's cash flow change in 2015?",
        "answer": {
            "answer": "The cash flow of BASF decreased significantly in 2015.",
            "verdict": 1,
        },
    },
    {
        "context": """{'Kennzahlen': 'Jahresueberschuss',
                '2012': 1526.0, '2013': 1625.0, '2014': 1662.0, '2015': 1968.0,
                '2016': 2093.0, '2017': 2541.0, '2018': 2330.0, '2019': 2103.0,
                '2020': 1424.0, '2021': 1629.0, '2022': 1253.0, '2023': 1340.0,
                'Unternehmen': 'Fresenius'}""",
        "question": "What was the year-end surplus of Fresenius in 2017?",
        "answer": {
            "answer": "The year-end surplus of Fresenius in 2017 was 2541",
            "verdict": 1,
        },
    },
]

multi_context_question_prompt_custom_examples = [
    {
        "question": "What was the net income of Commerzbank in 2023?",
        "context1": """{{'Kennzahlen': 'Jahresueberschuss',
                '2012': 4282.0, '2013': 4409.0, '2014': 5507.0, '2015': 7380.0,
                '2016': 5584.0, '2017': 6094.0, '2018': 6120.0, '2019': 5648.0,
                '2020': 4200.0, '2021': 6697.0, '2022': 4392.0, '2023': 8529.0,
                'Unternehmen': 'Commerzbank'}}""",
        "context2": """{{'Kennzahlen': 'Jahresueberschuss',
                '2012': 495.2, '2013': 653.0, '2014': 788.5, '2015': 649.0,
                '2016': 747.6, '2017': 896.0, '2018': 852.5, '2019': 1035.4,
                '2020': 1125.1, '2021': 1264.9, '2022': 1563.2, '2023': 1796.8,
                'Unternehmen': 'BASF'}}""",
        "output": "In 2023, by how much did Commerzbank’s net income exceed BASF’s net income?",
    },
    {
        "question": "How did Brenntag’s EBIT change from 2015 to 2016?",
        "context1": """{{'Kennzahlen': 'EBIT',
                '2012': 11498.0, '2013': 11671.0, '2014': 12697.0, '2015': -4069.0,
                '2016': 7103.0, '2017': 13818.0, '2018': 13920.0, '2019': 16960.0,
                '2020': 9675.0, '2021': 19275.0, '2022': 22124.0, '2023': 22576.0,
                'Unternehmen': 'Brenntag'}}""",
        "context2": """{{'Kennzahlen': 'EBIT',
                '2012': 6778.0, '2013': 5804.0, '2014': 7310.0, '2015': 7276.0,
                '2016': 7452.0, '2017': 7615.0, '2018': 6183.0, '2019': 6403.0,
                '2020': 4444.0, '2021': 6016.0, '2022': 7198.0, '2023': 10555.0,
                'Unternehmen': 'Commerzbank'}}""",
        "output": """Did Brenntag’s EBIT improve from 2015 to 2016 more than Commerzbank’s EBIT did,
                and what were the changes?""",
    },
]

multi_context_rewrite_invalid_question_custom_prompt = Prompt(
    name="rewrite_question",
    instruction=(
        """Given a context, question, and feedback, rewrite the question to improve
                its clarity and answerability based on the feedback provided."""
    ),
    output_format_instruction="",
    examples=[
        {
            "context": [
                """{'Kennzahlen': 'EBITDA-Marge',
                    '2012': 0.0, '2013': 0.0, '2014': 0.0, '2015': 0.0,
                    '2016': 0.0, '2017': 0.0, '2018': 8.91, '2019': 8.89,
                    '2020': 5.07, '2021': 11.36, '2022': 9.06, '2023': 11.3,
                    'Unternehmen': 'Siemens'}",
                    "{'Kennzahlen': 'EBITDA-Marge',
                    '2012': 0.0, '2013': 0.0, '2014': 0.0, '2015': 19.6,
                    '2016': 20.03, '2017': 20.5, '2018': 18.6, '2019': 20.11,
                    '2020': 19.34, '2021': 20.07, '2022': 19.66, '2023': 17.03,
                    'Unternehmen': 'Siemens Healthineers'}"""
            ],
            "question": "When did profitability surpass?",
            "feedback": (
                """The question is unclear because it doesn't specify whose profitability
                        is being referred to and which companies are being compared. Additionally,
                        it doesn't mention the metric used for profitability. To improve the
                        question, specify the companies involved and the metric."""
            ),
            "output": "When did Siemens' profitability surpass Siemens Healthineers'?",
        },
        {
            "context": [
                """{'Kennzahlen': 'Jahresueberschuss',
                    '2012': 1306.0, '2013': -2757.0, '2014': 1704.0, '2015': -170.0,
                    '2016': -5710.0, '2017': 1900.0, '2018': 1073.0, '2019': 9156.0,
                    '2020': 1054.0, '2021': 832.0, '2022': 2992.0, '2023': 1597.0,
                    'Unternehmen': 'Volkswagen VZ'}""",
                """{'Kennzahlen': 'Jahresueberschuss',
                    '2012': 64.25, '2013': 70.64, '2014': 68.44, '2015': 117.4,
                    '2016': 145.0, '2017': 159.33, '2018': 197.48, '2019': 218.74,
                    '2020': 299.56, '2021': 426.98, '2022': 913.1, '2023': 290.0,
                    'Unternehmen': 'Porsche VZ'}""",
            ],
            "question": "How many years did it outperform in surplus?",
            "feedback": (
                """The question is unclear because it doesn't specify which company 'it'
                        refers to, and which company it's being compared against. Also, the term
                        'surplus' could be more specific. To improve the question, specify the
                        companies involved and clarify the financial metric."""
            ),
            "output": """How many years did Volkswagen VZ outperform Porsche VZ
                    in terms of yearly surplus?""",
        },
        {
            "context": [
                """{'Kennzahlen': 'EBIT',
                    '2012': 3204.0, '2013': 3342.0, '2014': 3171.0, '2015': 3122.0,
                    '2016': 2581.0, '2017': 392.0, '2018': 2275.0, '2019': 2707.0,
                    '2020': 1211.0, '2021': 2932.0, '2022': 3419.0, '2023': 4597.0,
                    'Unternehmen': 'Airbus'}""",
                """{'Kennzahlen': 'EBIT',
                    '2012': 1762.0, '2013': 2211.0, '2014': 2177.0, '2015': 1719.0,
                    '2016': 2781.0, '2017': 2853.0, '2018': 2224.0, '2019': 2776.0,
                    '2020': 3176.0, '2021': 5423.0, '2022': 5717.0, '2023': 3935.0,
                    'Unternehmen': 'Mercedes-Benz Group'}""",
            ],
            "question": "Which year did EBIT top the other?",
            "feedback": (
                """The question is unclear because it doesn't specify which company's
                        EBIT is being referred to, which company it's being compared to, and
                        the specific metric. To improve the question, specify the companies
                        involved and clarify the financial metric."""
            ),
            "output": "Which year did Airbus' EBIT top Mercedes-Benz Group's?",
        },
    ],
    input_keys=["context", "question", "feedback"],
    output_key="output",
    output_type="str",
    language="english",
)


filter_question_custom_prompt = Prompt(
    name="filter_question",
    instruction="""\nAssess the given question for clarity and answerability, considering
            the following criteria:\n1. Clear Intent: Is it clear what type of answer or information
            the question seeks? The question should convey its purpose without ambiguity, allowing
            for a direct and relevant response.\n2. Specificity: Is the question specific enough to
            allow for a focused answer? Broad or overly general questions should be refined to
            target specific information.\n3. Appropriateness for Retrieval: Is the question suitable
            for retrieval augmented generation? That is, can it be answered using information that
            could be retrieved from a knowledge base or provided context?\nBased on these criteria,
            assign a verdict of **"1"** if the question is clear, specific, and appropriate for
            retrieval, making it understandable and answerable based on information that could be
            retrieved. Assign **"0"** if it fails to meet one or more of these criteria due to
            vagueness, ambiguity, or inappropriateness.Provide feedback and a verdict in **JSON**
            format, including suggestions for improvement if the question is deemed unclear.
            Highlight aspects of the question that contribute to its clarity or lack thereof,
            and offer advice on how it could be reframed or detailed for better understanding
            and answerability.""",
    output_format_instruction=get_json_format_instructions(QuestionFilter),
    examples=[
        {
            "question": "What was the EBIT of Brenntag in 2012?",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": """The question is clear, specific, and suitable for
                            retrieval. It asks for the EBIT of Brenntag in 2012.""",
                    "verdict": 1,
                }
            ).dict(),
        },
        {
            "question": "Tell me about the company.",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": """The question is too vague and lacks specificity.
                            It does not specify which company or what information is being
                            requested. To improve, specify the company and the information
                            needed.""",
                    "verdict": 0,
                }
            ).dict(),
        },
        {
            "question": "How does the profit in context compare to previous years?",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": """The question is unclear because it refers to 'context'
                            without specifying what the context is. To improve, specify the company,
                            the metric, and the years for comparison.""",
                    "verdict": 0,
                }
            ).dict(),
        },
        {
            "question": "How did RWE & E.ON perform financially in the same year?",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": """The question lacks specificity. It does not specify which
                            year to compare. To improve, specify the year""",
                    "verdict": 0,
                }
            ).dict(),
        },
        {
            "question": "Compare the financial performance.",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": """The question is too broad and lacks specificity. It does
                            not specify which companies or metrics to compare. To improve, specify
                            the companies and financial metrics of interest.""",
                    "verdict": 0,
                }
            ).dict(),
        },
    ],
    input_keys=["question"],
    output_key="output",
    output_type="json",
    language="english",
)
