"""Shared data for tests and benchmarks."""

TECHNICAL_ARCHITECTURE_QUESTION = """
Should a 50-person startup migrate from a monolithic Rails application to microservices?

Context:
- Current: Rails monolith serving 2M users
- Revenue: $5M ARR, growing 15% month-over-month
- Team: 10 backend engineers, 5 frontend, 2 DevOps
- Pain points:
  * Deployments take 2 hours
  * Test suite runs for 45 minutes
  * New features blocked by monolith coupling
- Runway: 18 months
- Competition: Two well-funded competitors moving faster

Requirements:
1. Analyze technical feasibility given current team and runway
2. Identify specific risks and mitigation strategies
3. Consider alternative approaches (partial migration, modular monolith, etc.)
4. Provide a concrete recommendation with timeline and first steps

This decision affects $2M+ in engineering costs over 18 months and could determine company survival.
""".strip()

POLICY_ANALYSIS_QUESTION = """
Design a remote work policy for a 200-person Series B SaaS company.

Stakeholders:
- Engineering: Wants full remote flexibility
- Sales: Prefers hybrid for collaboration
- Finance: Concerned about office lease costs ($400K/year)
- HR: Worried about culture and retention
- Execs: Want to optimize for productivity and growth

Balance competing priorities and identify potential issues.
""".strip()

STRATEGIC_BUSINESS_QUESTION = """
Should a B2B SaaS company with $10M ARR pursue enterprise upmarket expansion or SMB downmarket?

Current state:
- Mid-market focus ($50K-$500K ACV)
- Product: Mature, feature-complete
- Sales: 5 AEs, 2 SEs, 1-month sales cycle
- CAC: $15K, LTV: $180K, LTV/CAC = 12
- Churn: 8% annually

Evaluate market opportunity, execution risk, resource requirements, and timeline.
""".strip()


TEST_QUESTIONS = [
    {
        "domain": "technical_architecture",
        "question": TECHNICAL_ARCHITECTURE_QUESTION,
    },
    {
        "domain": "policy_analysis",
        "question": POLICY_ANALYSIS_QUESTION,
    },
    {
        "domain": "strategic_business",
        "question": STRATEGIC_BUSINESS_QUESTION,
    },
]
