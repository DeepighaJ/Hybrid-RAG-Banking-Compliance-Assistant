"""
Curated evaluation dataset for the RAG pipeline.

Each entry contains:
  - question: the query sent to the pipeline
  - reference: a short, hand-written ground-truth answer, used by
    Ragas' ContextPrecision metric to judge whether retrieval surfaced
    the right chunks (not just whether the answer sounded plausible)
  - query_type: informal label for grouping results when reviewing scores
    (not used by Ragas itself)

Source documents (fictional, self-authored):
  - Eastmere Bank KYC and CDD Procedures FAQ (EMB-KYC-FAQ-001), Q&A format
  - Eastmere Bank AML Risk Assessment Policy (EMB-AML-RAP-002), narrative

Query types:
  - single_document_faq / single_document_policy: answerable from one document
  - calculation: requires applying the RA-3.4 scoring formula stated in the policy
  - cross_document: requires combining facts from both documents
  - precision_test: near-duplicate facts exist elsewhere in the corpus (e.g. 25%
    vs 10% ownership thresholds, several different review frequencies), or the
    question targets an exact policy code that keyword search should catch
  - out_of_scope: not covered by either document; the pipeline must decline
"""

EVAL_DATASET = [
    # ------------------------------------------------------------------
    # Single document: KYC and CDD Procedures FAQ
    # ------------------------------------------------------------------
    {
        "question": "Can an account be opened before customer verification is complete?",
        "reference": (
            "Generally no. Verification must be completed before an account is opened. "
            "The only exception is Low-risk retail customers, whose verification may be "
            "completed within 10 business days of opening, with no outgoing payments "
            "permitted until it is complete. The exception never applies to Medium-risk "
            "or High-risk customers or to business customers (CDD-2.4)."
        ),
        "query_type": "single_document_faq",
    },
    {
        "question": "Who must approve a relationship with a politically exposed person?",
        "reference": (
            "The Head of Financial Crime Compliance, or a named deputy formally delegated "
            "in writing. Relationship managers and branch managers cannot approve PEP "
            "relationships (CDD-5.3)."
        ),
        "query_type": "single_document_faq",
    },
    {
        "question": "Which events trigger a customer review outside the periodic review cycle?",
        "reference": (
            "A trigger-based review must begin within 5 business days of: a change in "
            "ownership or control; credible adverse media; an unusual activity alert "
            "escalated to the Financial Intelligence Unit; the customer moving to a "
            "High-risk jurisdiction; or a request for a product outside the expected "
            "activity profile. A completed trigger-based review resets the periodic "
            "review date (CDD-6.3)."
        ),
        "query_type": "single_document_faq",
    },
    {
        "question": "How long must KYC and CDD records be retained?",
        "reference": (
            "For 7 years after the end of the customer relationship. Records relating to "
            "a suspicious activity report must be kept for 7 years from the date of the "
            "report or 7 years after the relationship ends, whichever is later (CDD-7.1)."
        ),
        "query_type": "single_document_faq",
    },
    # ------------------------------------------------------------------
    # Single document: AML Risk Assessment Policy
    # ------------------------------------------------------------------
    {
        "question": "What are the risk factors in the customer risk scoring model and how are they weighted?",
        "reference": (
            "Four factors: Customer Type (30%), Geography (30%), Products and Services (25%), and Delivery Channel (15%) (RA-3.2)."
        ),
        "query_type": "single_document_policy",
    },
    {
        "question": "Which customers are automatically rated High risk regardless of their score?",
        "reference": (
            "Politically exposed persons and their family members and close associates; "
            "all correspondent banking relationships; customers who were the subject of "
            "an internal suspicious activity report (Form EMB-ISR-01) in the previous 24 "
            "months; and customers with confirmed, credible adverse media relating to "
            "financial crime (RA-3.7)."
        ),
        "query_type": "single_document_policy",
    },
    {
        "question": "What happens to customer relationships connected to a jurisdiction rated Prohibited?",
        "reference": (
            "They must be exited within 60 calendar days, unless the MLRO approves a "
            "longer period to allow an orderly exit (RA-4.4)."
        ),
        "query_type": "single_document_policy",
    },
    # ------------------------------------------------------------------
    # Calculation: applying the RA-3.4 formula
    # ------------------------------------------------------------------
    {
        "question": (
            "A trust connected to a Medium-rated jurisdiction uses private banking "
            "services and was onboarded face-to-face. What is its weighted risk score "
            "and risk tier?"
        ),
        "reference": (
            "Customer Type 4 x 0.30 = 1.20, Geography 3 x 0.30 = 0.90, Products and "
            "Services 4 x 0.25 = 1.00, Delivery Channel 1 x 0.15 = 0.15. The weighted "
            "risk score is 3.25, which is Medium risk (RA-3.3 to RA-3.5)."
        ),
        "query_type": "calculation",
    },
    {
        "question": (
            "A salaried individual opens a basic current account through digital "
            "onboarding with live video verification, but their geography data is "
            "missing. What is their weighted risk score and risk tier?"
        ),
        "reference": (
            "Missing risk factor data is scored 5, the highest score (RA-6.3). Customer "
            "Type 1 x 0.30 = 0.30, Geography 5 x 0.30 = 1.50, Products and Services "
            "1 x 0.25 = 0.25, Delivery Channel 2 x 0.15 = 0.30. The weighted risk score "
            "is 2.35, which is Medium risk (RA-3.5)."
        ),
        "query_type": "calculation",
    },
    # ------------------------------------------------------------------
    # Cross document: needs both the policy and the FAQ
    # ------------------------------------------------------------------
    {
        "question": (
            "What weighted risk score makes a customer High risk, and how often must a "
            "High-risk customer be reviewed?"
        ),
        "reference": (
            "A weighted risk score of 3.50 or above is High risk (RA-3.5). High-risk "
            "customers must be reviewed every 12 months (CDD-6.1)."
        ),
        "query_type": "cross_document",
    },
    {
        "question": (
            "For the company in the policy's worked example, which beneficial ownership "
            "threshold applies?"
        ),
        "reference": (
            "The worked example company is rated High risk (RA-3.6), so the 10% beneficial ownership threshold for High-risk business customers applies (CDD-4.2)."
        ),
        "query_type": "cross_document",
    },
    {
        "question": (
            "If a country's rating is raised to High, what must happen to customers "
            "connected to that country, and how quickly?"
        ),
        "reference": (
            "Every connected customer must undergo a trigger-based review and have their "
            "risk score recalculated (RA-4.4). A trigger-based review must begin within "
            "5 business days (CDD-6.3)."
        ),
        "query_type": "cross_document",
    },
    {
        "question": (
            "What risk tier does a politically exposed person receive, who approves the "
            "relationship, and how often is it reviewed?"
        ),
        "reference": (
            "PEPs are automatically rated High risk (RA-3.7). The relationship must be "
            "approved by the Head of Financial Crime Compliance or a formally delegated "
            "deputy (CDD-5.3), and High-risk customers are reviewed every 12 months "
            "(CDD-6.1)."
        ),
        "query_type": "cross_document",
    },
    # ------------------------------------------------------------------
    # Precision tests: near-duplicate facts and exact policy codes
    # ------------------------------------------------------------------
    {
        "question": "What is the beneficial ownership threshold for Medium-risk business customers?",
        "reference": (
            "25% or more of the shares or voting rights, directly or indirectly (CDD-4.1)."
        ),
        "query_type": "precision_test",
    },
    {
        "question": "How often is the Country Risk List reviewed?",
        "reference": (
            "Quarterly, and it must also be updated within 10 business days of any "
            "relevant change to international monitoring lists or sanctions regimes "
            "(RA-4.3)."
        ),
        "query_type": "precision_test",
    },
    {
        "question": "Who approves changes to the AML Risk Assessment Policy?",
        "reference": (
            "The Board Risk Committee (RA-1.4)."
        ),
        "query_type": "precision_test",
    },
    {
        "question": "What does policy CDD-5.4 say?",
        "reference": (
            "A customer remains classified as a politically exposed person for at least "
            "24 months after leaving the public function. Declassification after that is "
            "not automatic and must be approved by the MLRO based on a documented risk "
            "assessment (CDD-5.4)."
        ),
        "query_type": "precision_test",
    },
    # ------------------------------------------------------------------
    # Out of scope: the pipeline must decline
    # ------------------------------------------------------------------
    {
        "question": "What is the minimum credit score required to open a business account?",
        "reference": (
            "This information is not contained in either source document; the "
            "pipeline should decline to answer rather than fabricate a figure."
        ),
        "query_type": "out_of_scope",
    },
    {
        "question": "What interest rate does Eastmere Bank pay on savings accounts?",
        "reference": (
            "This information is not contained in either source document; the "
            "pipeline should decline to answer rather than fabricate a figure."
        ),
        "query_type": "out_of_scope",
    },
]