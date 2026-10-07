template_a = """{name}

PERSONAL STATEMENT
------------------
Mathematics undergraduate with strong analytical skills,
seeking opportunities to apply mathematical modelling and data analysis
in real-world problem-solving. Enthusiastic about research and software development.

EDUCATION
---------
{university}                             Oct 2022 - Present
BA Mathematics (expected 1st Class Honours)

{school}, {school_location}              Sept 2017 - June 2022
A-Levels: {a_levels}

SKILLS
------
- Programming: Python, R, MATLAB
- Mathematical techniques: Linear algebra, Calculus, Probability & Statistics
- Data analysis & visualization: Pandas, Matplotlib, Excel
- Strong problem-solving and logical reasoning

PROJECTS
--------
Data Analysis of UK Climate Trends            Mar 2024 - May 2024
- Analysed historical temperature and rainfall data using Python
- Created predictive models to assess future climate scenarios

Mathematical Modelling of Disease Spread      Jan 2023 - Apr 2023
- Developed SIR models to simulate infection dynamics
- Used differential equations and numerical methods

WORK EXPERIENCE
---------------
Intern, Data Science Team, {company}     July 2023 - Aug 2023
- Assisted in cleaning and analysing large datasets
- Automated data reporting scripts using Python

VOLUNTEERING
------------
Math Tutor, Local Community Centre            Sept 2021 - Present
- Support GCSE students with exam preparation and concept understanding

INTERESTS
---------
Chess, hiking, and mathematical puzzles
"""


template_b = """{name}

PERSONAL STATEMENT
------------------
Computer Science undergraduate with strong programming skills,
passionate about software engineering and building scalable systems.
Looking to apply technical expertise in a fast-paced development environment.

EDUCATION
---------
{university}                             Oct 2022 - Present
BSc Computer Science (expected 1st Class Honours)

{school}, {school_location}              Sept 2017 - June 2022
A-Levels: {a_levels}

SKILLS
------
- Languages: Python, Java, TypeScript, C++
- Web development: React, Node.js, FastAPI
- Databases: PostgreSQL, MongoDB, Redis
- Tools: Git, Docker, AWS, CI/CD pipelines
- Strong algorithmic thinking and debugging

PROJECTS
--------
Full-Stack Task Management App                Mar 2024 - May 2024
- Built a React frontend with a FastAPI backend and PostgreSQL database
- Implemented real-time notifications using WebSockets

Distributed File Storage System                Jan 2023 - Apr 2023
- Designed a fault-tolerant file storage system using Go and gRPC
- Achieved 99.9% availability through replication strategies

WORK EXPERIENCE
---------------
Software Engineering Intern, {company}   June 2023 - Sept 2023
- Developed RESTful APIs serving 10k+ daily requests
- Wrote unit and integration tests improving code coverage by 30%

VOLUNTEERING
------------
Code Club Mentor, Local Secondary School      Oct 2022 - Present
- Teach Python basics to students aged 11-14

INTERESTS
---------
Open source contribution, hackathons, running
"""


template_c = """{name}

PERSONAL STATEMENT
------------------
Economics undergraduate with strong quantitative and analytical abilities,
seeking to apply economic modelling and financial analysis in a business environment.
Detail-oriented with excellent communication and presentation skills.

EDUCATION
---------
{university}                             Oct 2022 - Present
BSc Economics (expected 1st Class Honours)

{school}, {school_location}              Sept 2017 - June 2022
A-Levels: {a_levels}

SKILLS
------
- Statistical software: Stata, R, SPSS, Excel
- Financial analysis: Financial modelling, Valuation, Forecasting
- Data tools: SQL, Tableau, Bloomberg Terminal basics
- Strong written and verbal communication
- Team collaboration and project management

PROJECTS
--------
Analysis of UK Inflation Drivers               Mar 2024 - May 2024
- Modelled inflation trends using time-series econometrics
- Presented findings to a panel of economics lecturers

Market Structure and Pricing Strategies        Jan 2023 - Apr 2023
- Analysed oligopolistic behaviour in the telecoms sector
- Wrote a research paper on price discrimination effects

WORK EXPERIENCE
---------------
Finance Intern, {company}                 July 2023 - Aug 2023
- Assisted in preparing quarterly financial reports
- Conducted competitor benchmarking analysis

VOLUNTEERING
------------
Economics Peer Tutor, University Society       Oct 2022 - Present
- Help first-year students with microeconomic theory

INTERESTS
---------
Investment reading, debating, cycling
"""


template_d = """{name}

PERSONAL STATEMENT
------------------
Undergraduate student exploring career options in business and analytics.
Hardworking and keen to develop professional skills through hands-on experience.

EDUCATION
---------
{university}                             Oct 2022 - Present
BA Business Studies (expected 2:1)

{school}, {school_location}              Sept 2017 - June 2022
A-Levels: {a_levels}

SKILLS
------
- Microsoft Office: Word, Excel, PowerPoint
- Basic Python programming
- Social media management
- Teamwork and interpersonal skills

PROJECTS
--------
University Societies Budget Review             Mar 2024 - Apr 2024
- Helped track society spending across the academic year
- Created a simple spreadsheet for budget allocation

Retail Customer Survey Analysis                Jan 2023 - Mar 2023
- Collected and summarised survey responses from 50 participants
- Presented results using PowerPoint charts

WORK EXPERIENCE
---------------
Retail Assistant, {company}               June 2023 - Aug 2023
- Operated the till and handled customer queries
- Assisted with stock management and visual merchandising

VOLUNTEERING
------------
Charity Shop Volunteer                          Oct 2022 - Present
- Sort donations and assist customers on weekends

INTERESTS
---------
Social media, fashion, watching documentaries
"""


template_e = """{name}

PERSONAL STATEMENT
------------------
STEM undergraduate with a strong track record in research and publication.
Highly motivated to apply advanced analytical skills to challenging problems
in a research-intensive environment. Experienced in experimental design.

EDUCATION
---------
{university}                             Oct 2022 - Present
BSc Physics (expected 1st Class Honours)

{school}, {school_location}              Sept 2017 - June 2022
A-Levels: {a_levels}

SKILLS
------
- Programming: Python, C++, Julia, Wolfram Mathematica
- Research: Experimental design, Statistical analysis, Literature review
- Scientific computing: NumPy, SciPy, TensorFlow, COMSOL
- Technical writing: LaTeX, academic publishing standards
- Data visualisation: Matplotlib, Plotly, OriginPro

PUBLICATIONS
------------
"Damped Harmonic Oscillations in Non-Newtonian Fluids"
Undergraduate Research Journal, 2025, Vol. 12, pp. 45-58
- Co-authored experimental study on fluid dynamics
- Designed data acquisition and analysis pipeline in Python

"Machine Learning for Particle Track Reconstruction"
Conference on Undergraduate Research in Physics, Mar 2024
- Poster presentation on ML-based classification methods

PROJECTS
--------
Quantum Error Correction Simulation            Oct 2024 - Present
- Implementing surface codes using Qiskit and Python
- Simulating error rates under different noise models

Thermal Conductivity of Aerogels               Jan 2024 - Apr 2024
- Built a custom apparatus to measure thermal properties
- Analysed results using finite element modelling

WORK EXPERIENCE
---------------
Research Intern, {company}                June 2024 - Sept 2024
- Assisted in developing Monte Carlo simulations for particle physics
- Processed experimental data and contributed to a published paper

VOLUNTEERING
------------
STEM Ambassador, Local Schools                  Oct 2022 - Present
- Deliver interactive science demonstrations to secondary students

INTERESTS
---------
Electronics, 3D printing, rock climbing
"""


templates = {
    "template_a": template_a,
    "template_b": template_b,
    "template_c": template_c,
    "template_d": template_d,
    "template_e": template_e,
}
