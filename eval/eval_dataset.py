"""
Curated evaluation dataset for the RAG pipeline.

Each entry contains:
  - question: the query sent to the pipeline
  - reference: a short, hand-written ground-truth answer, used by
    Ragas' ContextPrecision metric to judge whether retrieval surfaced
    the right chunks (not just whether the answer sounded plausible)
  - query_type: informal label for grouping results when reviewing scores
    (not used by Ragas itself)

Questions span both source documents individually and, for the
cross_document entries, require synthesizing information from both —
useful for spotting whether hybrid retrieval is pulling from the correct
source(s) rather than defaulting to whichever document dominates the index.
"""

EVAL_DATASET = [
    {
        "question": "How often should country risk data sources be refreshed?",
        "reference": (
            "As a general rule, ratings should be updated no less frequently than "
            "annually, following updates of the underlying data sources. Interim "
            "updates may also be triggered by key events such as FATF public "
            "statements, new sanctions, or regime change."
        ),
        "query_type": "single_document_faq",
    },
    {
        "question": "What is Financial Crime Country Risk (FCCR)?",
        "reference": (
            "FCCR is the term used to describe the residual level of financial "
            "crime risk in a country after considering both the inherent risk "
            "and the effectiveness of that country's AML/CTF framework."
        ),
        "query_type": "single_document_faq",
    },
    {
        "question": "How should FIs handle missing data points when assessing country risk?",
        "reference": (
            "FIs may substitute missing values with the highest risk level, lowest "
            "risk level, an average score, or a proxy value, and should perform a "
            "data quality assessment when data is missing. A subjective override "
            "should be considered where information is particularly limited, and "
            "any defaulting decisions must be clearly documented."
        ),
        "query_type": "single_document_faq",
    },
    {
        "question": "Should overrides or discretionary changes to country risk ratings be allowed?",
        "reference": (
            "Overrides should be very limited and subject to a stringent "
            "justification and governance process, since frequent overrides can "
            "undermine the integrity of the risk methodology. Approved exceptions "
            "should have an expiration date and be reviewed at least annually."
        ),
        "query_type": "single_document_faq",
    },
    {
        "question": "Who should own the FCCR methodology within a financial institution?",
        "reference": (
            "The FCCR methodology should be owned centrally by a group-level unit "
            "independent from the business, typically an independent financial "
            "crime compliance, AML, or risk function, though day-to-day operational "
            "maintenance may be delegated to another unit that reports into that owner."
        ),
        "query_type": "single_document_faq",
    },
    {
        "question": "What are the three key elements of a risk-based approach according to the Wolfsberg Group?",
        "reference": (
            "The three key elements are proportionality, prioritisation, and "
            "effectiveness."
        ),
        "query_type": "single_document_rba",
    },
    {
        "question": "What does proportionality mean in the context of a financial crime risk management programme?",
        "reference": (
            "Proportionality means a financial institution's FCRM programme should "
            "be proportionate to its business model, as determined by its size, "
            "scale, footprint, customers, and risk appetite, rather than applying "
            "a one-size-fits-all approach."
        ),
        "query_type": "single_document_rba",
    },
    {
        "question": "What customer-related risk variables should FIs consider when prioritising higher-risk customers?",
        "reference": (
            "Risk variables include customer type, industry/business type, "
            "customer legal structure, whether the customer is publicly owned or "
            "traded, and whether the customer is resident, incorporated, or does "
            "significant business in higher-risk countries."
        ),
        "query_type": "single_document_rba",
    },
    {
        "question": "What are the Wolfsberg Factors that define an effective FCRM programme?",
        "reference": (
            "The Wolfsberg Factors are: complying with AML/CTF laws and "
            "regulations, establishing a reasonable and risk-based set of controls "
            "to mitigate the risk of being used to facilitate illicit activity, and "
            "providing highly useful information to relevant government agencies "
            "in defined priority areas."
        ),
        "query_type": "single_document_rba",
    },
    {
        "question": (
            "How does country risk factor into both the FCCR methodology and the "
            "risk-based approach's prioritisation principle?"
        ),
        "reference": (
            "Country risk is one input into an FI's FCCR rating methodology, which "
            "assesses the residual financial crime risk of a country. Under the "
            "risk-based approach, prioritisation uses country risk exposure "
            "(e.g. a customer resident, incorporated, or doing significant business "
            "in a higher-risk country) as one factor in deciding which customers "
            "and activities to focus resources and enhanced due diligence on."
        ),
        "query_type": "cross_document",
    },
    {
        "question": (
            "How do sanctions considerations relate to both the country risk "
            "methodology and the effectiveness principle of the risk-based approach?"
        ),
        "reference": (
            "FIs generally default sanctioned regimes to the highest risk rating in "
            "their country risk methodology, applying the highest levels of due "
            "diligence. This connects to effectiveness in the risk-based approach, "
            "which calls for a reasonable, risk-based set of controls to mitigate "
            "the risk of facilitating illicit activity, including sanctions evasion."
        ),
        "query_type": "cross_document",
    },
    {
        "question": "Who owns the FCCR methodology?",
        "reference": (
            "The FCCR methodology should be owned centrally by a group-level unit "
            "independent from the business, such as an independent financial crime "
            "compliance, AML, or risk function."
        ),
        "query_type": "precision_test",
    },
    {
        "question": "Who are the users of FCCR ratings, and how are they disseminated?",
        "reference": (
            "Users include Lines of Business, and teams involved in Customer Due "
            "Diligence, transaction monitoring, and Enterprise Wide Risk "
            "Assessments. Once approved by governance, ratings are provided to "
            "relevant stakeholders and reference data management teams for "
            "implementation within a set timescale."
        ),
        "query_type": "precision_test",
    },
    {
        "question": "What is the minimum credit score required to open a business account?",
        "reference": (
            "This information is not contained in either source document; the "
            "pipeline should decline to answer rather than fabricate a figure."
        ),
        "query_type": "out_of_scope",
    },
]
