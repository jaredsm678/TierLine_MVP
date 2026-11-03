MVP of my LLM vs S1 analysis on different difficulty inputs


Thesis
1. System One models underperform frontier LLMs on difficult inputs at a much higher degree than on easy inputs
    1. There is lots of evidence of this
        1. See jevs own tests: https://evals.typesafe.ai/
        2. IBM difficulty paper: https://aclanthology.org/2021.naacl-industry.38/
        3. 2024 Paper comparing LLMs and trained classifiers: https://arxiv.org/abs/2608.20371
        4. Easy2Hard-Bench: https://arxiv.org/abs/2409.18433
2. A general difficulty definition computable from input at inference time alone that demonstrates the same pattern of higher difficulty -> higher underperformance exists
    1. This doesn’t exist yet we need to make this. Its possible we can use a variation of the ibm difficulty definition to apply to general datasets
3. There is an optimal difficulty boundary where we can decide which model to use for an input such that the combined use of an S1 model and an LLM outperforms the S1 on accuracy significantly and the LLM on cost significantly
    1. Prior research that does similar routing for quality LLM -> cheaper LLM but based on model uncertainty: https://aclanthology.org/2024.emnlp-industry.114/
    2. This is the main goal of the research
    3. This claim is heavily dependent on claim 2 - If we can prove claim 2, the implementation of this should be very straightforward
MVP
- The MVP only attempts to address claim 1
- The MVP has statistically significant proof for claim 1, but so do other papers

Purpose
- Create an LLM evaluation because its often cited as something ai engineering roles look for as experience/side project
- We are submitting this research to this competition: https://www.kaggle.com/competitions/gemma-4-developer-agent-paper 
- This research should be evaluated based on the 5 evaluation categories in the competition
    - Novelty: Does the work provide new insights, deepen understanding, or highlight important properties of existing methods?
    - Quality: How general and universal is the Submission approach beyond the competition? Does your submission translate to other similar problems? How well does your method generalize?
    - Relevance: Is there significance and/or potential impact of the research on the field of software engineering and agentic learning?
    - Verifiability: Does the paper include sufficient information to understand how an innovation works; to understand how data was obtained, analyzed, and interpreted?
    - Clarity: Is the paper clearly presented and written?
MVP
- The MVP alone is heavily lacking on novelty, quality, and relevance
- If I am able to prove and develop methodology for claim 2 and 3 I think the research will be very strong on novelty, quality, and relevance

Dataset Selection
MVP
- We use the CLINIC150 dataset
- This dataset has heavy usage in classification and AI evaluation (Over 850+ citations)
    - Talk about training exposure issue
    - The 2024 LLM comparison paper also used this dataset
- Since there are 150 labels + OOS we choose to use one domain (banking) and simply provide the models with the 15 label definitions plus out of scope. We only give the model data with that has an answer as one of the 15 labels
￼<img width="715" height="415" alt="bnk-0006" src="https://github.com/user-attachments/assets/8ae7bdb8-d021-45c3-8a5b-af91e7564438" />
- This initial analysis lacks all the labels so we don’t do any analysis on out of scope predictions
Final Project
- The general difficulty definition should apply to many different datasets so we will likely use more than just the CLINIC150 dataset

Difficulty Definition
MVP
- The following paper defined difficulty that we used for the mvp
    - https://aclanthology.org/2021.naacl-industry.38/ 
    - Here is a more detailed explanation:
<img width="677" height="439" alt="The recipe, step by step" src="https://github.com/user-attachments/assets/e80c8be3-2a69-46d8-b88c-cc7bfc5dfa53" />
Final Project
- In order to prove our general difficulty definition works we will need to use multiple datasets as mentioned before
- This MVP difficulty definition is based on taking the worst instances of a label and defining them as difficult
    - This definition really only works for a training set. Real life tasks won’t really allow us to use this difficulty definition so we will need to develop a better difficulty definition for claim 2
- We need to find a way to get input difficulty in a way that’s generalizable and doesn’t depend on training data

Label Definitions
MVP
- Here are my label definitions
<img width="582" height="698" alt="- account_blocked The customer reports that their account is blocked," src="https://github.com/user-attachments/assets/866c1abd-f8bb-4c4f-a5ec-dbb58b442fd1" />

- We used the following paper to inform our model definitions: https://aclanthology.org/2024.sigdial-1.39/ 
    - We used these label definitions for both the LLMs and the S1s
    - This label definition framework is optimized for LLMs so that may bias the S1 model performance down
Final Project
- Maybe we need to consider a framework for label definitions that is more fair for LLMs and S1s
- Maybe we should have variations of label definitions so we can test llm optimized vs s1 optimized vs balanced?

Prompt Construction
MVP
- Paste prompt construction here
￼<img width="574" height="292" alt="You will route one customer request to the correct handler" src="https://github.com/user-attachments/assets/3cef2bb2-5c72-47f0-82a9-fe41d4386c13" />
- For cost purposes we did not vary this prompt
Final Project
- We should probably vary our prompt at some point? The original mvp results seemed reasonable so I didn’t see a need to vary it for the mvp
- Not sure how much of an effect varying this prompt will have on performance. Its likely small so it seems a low priority

Model Selection
MVP
- The LLMs were selected based on the reasoning performance of this benchmark: https://livebench.ai/#/ 
    - I only wanted one model per lab and I wanted to use the value choice if performance was quite similar (thus sol over astra and opus over fable). 
    - I used a similar method to select the S1 models.
Final Project
- We should ideally include as many as possible
    - The live bench link is a good place to get a list of models to use

Harness Creation
MVP
- Vibe coded this but I don’t think that’s such a big deal because all the other sections of this report were clearly defined before creating it 
Final Project
- After reading over all the code, I think some parts of it are slightly overcomplicated/lack readability. I think some time spent manually rewriting it or a heavier audit may be beneficial but im I think focusing on the difficulty metrics or dataset selections are a much higher priority

Results
MVP
<img width="806" height="620" alt="Claude 5 5 Opus" src="https://github.com/user-attachments/assets/197cc509-e5aa-446f-9f85-2e0db4fa2799" />
Final Results
- Results will need to have much more analysis in the final version
- Its hard to forecast too much on what we will analyze before developing the other sections more

Conclusion
- The mvp gave great evidence that claim 1 is likely correct and helped us to create simple momentum/findings on other sections. 
- I think we have a solid foundation to build upon further so I don’t think my work so far has been unproductive
- If we want to be competitive in the competition and contribute findings that other people outside the competition find relevant, we really need to work towards proving and providing methodology for claim 2 and 3 of the thesis
