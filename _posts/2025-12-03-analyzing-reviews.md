---
layout: post
title: "Analyzing Paper Reviews with LLMs: I Used ChatGPT, DeepSeek, Qwen, Mistral, Gemini, and Claude (and you should too + publish the analysis)"
description: "In the age of LLMs, leveraging their capabilities to analyze and interpret paper reviews can provide valuable insights. Here's how I used multiple LLMs to dissect the reviews I received for my recent submission."
date: 2025-12-03
categories: [R, Python, Misc]
comments: true
---

What I'd recommend you to do, as LLMs are becoming increasingly better at this, is to use them to help you analyze the reviews you received for your papers (increasingly bogus, and some people don't even bother to read). Thinking about doing it for the last reviews I received, not to say that I'm right, but for everyone to see [how _well_ I'm bein treated](https://www.change.org/stop_torturing_T_Moudiki). And I'm not the type to whine.

I did it for the framework described in [https://thierrymoudiki.github.io/blog/2025/10/19/r/python/P-Y-GAN-like](https://thierrymoudiki.github.io/blog/2025/10/19/r/python/P-Y-GAN-like), submitted to [https://world-model-mila.github.io/](https://world-model-mila.github.io/). 

**Hopefully LLMs will serve as reviewers soon, because IMHO, they'd be less biased and more objective than humans on this specific task.**

Here is an analysis of the reviews I received, **done with ChatGPT, DeepSeek, Qwen, Mistral, Gemini, and Claude** (you judge for yourself):

# ChatGPT

```markdown
Below is a structured, **neutral, point-by-point fairness assessment** of the four reviews you received for your DISTROSIMULATOR paper.
I evaluate each review according to:
**(1) Topic-fit accuracy, (2) Technical correctness, (3) Understanding of your method, (4) Tone & constructiveness, (5) Evidence of bias or misunderstanding.**

This is **not** an appeal letter—just an analysis of fairness.

---

# ⭐ Overall Summary

Across the four reviews:

* **Two reviewers (DT7u and dsDV)** appear **fair, careful, and constructive**, even if critical.
* **Two reviews (PSoS and tohC)** contain **major technical misunderstandings**, incorrect assumptions, or dismissive assessments that do not reflect the content of your paper.

The strongest issue across negative reviews is **misunderstanding the problem setup**, particularly:

### ❗ Incorrect assumption: *x and y are independent*

Reviewers PSoS and tohC assume that since **X is Gaussian noise**, it is *independent of Y*, so the optimal mapping is constant.
This is not true.
Your formulation *defines* a joint training set by pairing noise with targets as a *learned transport map*. The pairing is *arbitrary but consistent*, and the surrogate learns a function *only because θ is optimized by a distributional objective*.

They misinterpret X as an exogenous explanatory variable, not as latent noise.
This mistake leads them to conclude the method is trivial or wrong.

Because their core criticism is based on a false premise, **those reviews are factually incorrect**.

---

# ⭐ Review-by-Review Analysis

---

# 1. Reviewer DT7u — **Fair, balanced, technically engaged**

### ✔ Strengths of the review

* Correctly describes the method.
* Identifies real weaknesses (e.g., lacking ablations, needing more surrogate types).
* Makes reasonable suggestions (clarify variables, evaluate component contribution).
* Recognizes coherence and mathematical rigor.

### ✔ Fairness

**High.**
The reviewer understood the method, evaluated it reasonably, and provided actionable suggestions.
Even their reservations about world-model relevance are reasonable given your short discussion.

### ✔ Where the review might be slightly off

* They argue that supervised learning is an “inverse’’ of your mapping—but in fact your surrogate is not predicting labels from data but approximating a transport map.
* But this is a subtle conceptual distinction, not a factual error.

### ✔ Verdict

This is a **fair, thoughtful review** and not biased.
It correctly identifies gaps you could strengthen in a revision.

---

# 2. Reviewer PSoS — **Unfair due to major technical misunderstanding**

This is the most problematic review.

### ❗ Fundamental error

> “The noises x are sampled i.i.d. … y and x are independent, so E[y | x] = E[y]. Therefore f*(x) is constant.”

This is **incorrect**.

Why?

* In your algorithm, **X is not sampled independently per training sample** after pairing.
* You generate a *fixed* latent variable for each training datapoint (noise sample ↔ data sample).
* You then **optimize θ to minimize MMD(Y, fθ(X) + ε)**.
* The model does *not* attempt to estimate E[Y|X]; that is the regression objective, but the regression parameters are searched via *distribution matching*, not supervised risk minimization.
* Thus the pairing is part of a **learned transport**, not a regression dataset reflecting statistical causality.

This mistaken assumption invalidates 80–90% of their criticism.

### ❗ Additional fairness issues

* Calling the method “trivial” is opinion-based and dismissive.
* Topic-fit “poor” is questionable: your paper explicitly discusses world-modeling applications.

### ✔ Tone: Harsh and dismissive

The wording (“trivial”, “no choice but to reject”) is unusually negative for a workshop.

### ✔ Verdict

**This review is technically incorrect and unfair.**
It is based on a misinterpretation of your entire framework.

---

# 3. Reviewer dsDV — **Fair, positive, clearly understood the method**

### ✔ Strengths

* Correctly identifies your decomposition (conditional mean + residual noise).
* Appreciates stability, simplicity, and CPU efficiency.
* Correctly states limitations (scalability, PCA usage, time-series assumptions).
* Gives a positive but not blindly enthusiastic rating.

### ✔ Fairness

**High fairness.**
This reviewer understood your intent and evaluated the contribution appropriately for a workshop.

### ✔ Verdict

**A fair and accurate review**, aligned with the content of the paper.

---

# 4. Reviewer tohC — **Partly fair, partly incorrect, somewhat superficial**

### ✔ Some valid points

* Calls out clarity issues in the abstract and notation (these are valid).
* Notes that PCA-reduced MNIST is simplistic (true but acceptable for a workshop).
* Questions relevance to world-modeling (arguable but not unfair).

### ❗ Technical misunderstanding

They echo the same mistake as Reviewer PSoS:

> “What is X? What is Y? Notations unclear.”

This indicates a failure to understand the setup, even though Appendix + Section 2.1 explain it.

They conclude that because the novelty is unclear *to them*, the contribution is weak—which is a **subjective but unfair consequence of misunderstanding**.

### ❗ Topic-fit marked “Poor”

This is not accurate:
Your paper discusses generative transition models, world-modeling relevance, model-based RL.

At worst, topic fit should be “Fair,” not “Poor”.

### ✔ Tone: Neutral, but confused

Less harsh than PSoS but also less technically grounded.

### ✔ Verdict

**Partially unfair due to misunderstandings, but not malicious.**
This reviewer struggled with clarity and did not deeply engage with the method.

---

# ⭐ Overall Fairness Score Across Reviews

| Reviewer | Understanding | Technical Accuracy | Tone       | Fairness          |
| -------- | ------------- | ------------------ | ---------- | ----------------- |
| DT7u     | Good          | Good               | Respectful | **Fair**          |
| PSoS     | Poor          | ❌ Incorrect        | Dismissive | **Unfair**        |
| dsDV     | Excellent     | Accurate           | Positive   | **Fair**          |
| tohC     | Weak          | ❌ Some errors      | Neutral    | **Partly unfair** |

**2 fair reviews, 1 accidental misconception, 1 strongly incorrect/unfair review.**

---

# ⭐ Final Assessment

Your submission received a **mixed but skewed** set of reviews:

* **The positive review (dsDV) is fully grounded and credible.**
* **The neutral–critical review (DT7u) is fair and constructive.**
* **The two negative reviews (PSoS and tohC) are built on misunderstandings of both the methodology and the role of the noise variable X.**

  * PSoS’s review is *factually incorrect* and therefore unfair.
  * tohC’s review is *shallow and confused*, leading to an unfairly low assessment.

Given the novelty of your formulation, misunderstanding is not surprising—but **misunderstanding is not the same as methodological flaw**, and the strongest criticisms incorrectly treat it as such.

---

If you would like, I can:

✅ Help you **draft a rebuttal**
✅ Produce a **formal reviewer-response PDF**
✅ Provide a **clear explanation of why X and Y are *not* treated as independent**
✅ Suggest **revisions to improve clarity for future submissions**

Just tell me what you want next.
```

# DeepSeek 

```markdown
Given this is a **4-page workshop submission**, the expectations for completeness, benchmarking, and immediate applicability are generally lower than for a full conference paper. Workshop submissions often prioritize **novel ideas, conceptual frameworks, and community discussion potential**.  

Here is a re-examination of the reviews with this **4-page workshop context** in mind:

---

## ✅ **Fair Review** (Appropriate for a workshop)
**Reviewer dsDV (Rating: 4 - Accept)**  
- Recognizes the framework's novelty, computational efficiency, and relevance to world modeling.  
- Acknowledges limitations but sees value in the idea for workshop discussion.  
- **Fair for a workshop:** Engages with the vision, not just the current experiments.

**Reviewer DT7u (Rating: 2 - Weak Reject)**  
- Provides constructive feedback: test more surrogates, clarify notation, justify benefits.  
- Suggests useful experiments (ablation, neural surrogate test).  
- **Fair for a workshop:** Points to gaps but does not dismiss the idea outright; encourages improvement.

---

## ⚠️ **Overly Harsh / Unfair Reviews** (Given workshop norms)
**Reviewer PSoS (Rating: 1 - Strong Reject)**  
- Assumes statistical triviality due to independence of \(X\) and \(Y\), but misunderstands the *generator mapping* objective.  
- Dismisses the work as “trivial” despite shown results.  
- **Unfair for a workshop:** Even if the concern is valid, the tone is dismissive, and the criticism overlooks the paper’s stated aim—a framework proposal, not a statistical estimation method.

**Reviewer tohC (Rating: 1 - Strong Reject)**  
- Criticizes clarity and topic fit, though the paper explicitly mentions world modeling applications.  
- Calls contribution insufficient without engaging with the framework’s potential.  
- **Unfair for a workshop:** Workshop submissions are meant to foster discussion; rejecting outright for “unconvincing relevance” stifles that.

---

## 🧠 **Key Takeaways for a 4-Page Workshop Submission**
1. **Clarity is critical** — in short format, every definition and notation must be explicit early on.
2. **Workshop relevance must be front-and-center** — even briefly motivate how this helps world modeling (e.g., stable generative transitions for MBRL).
3. **It’s okay to be preliminary** — but you must acknowledge limitations and invite community exploration (which the paper does).
4. **Some reviewers will hold workshop papers to conference standards** — this is unfair but common. A strong rebuttal can address misunderstandings.

---

## 📝 **Suggested Rebuttal Strategy for the Authors**
- **To Reviewer PSoS:** Clarify that \(X\) is not a statistical covariate but a *latent noise input* to the generator; the surrogate learns a mapping, not a conditional expectation in the data space. Cite the non-constant outputs in Table 1 as evidence.
- **To Reviewer DT7u:** Acknowledge the need for more surrogate testing and commit to exploring neural networks in future work. Offer to add a short ablation in the camera-ready if accepted.
- **To Reviewer tohC:** Reiterate the workshop’s focus on generative world modeling and note that the framework’s stability and speed are direct benefits for model-based RL research.
- **To All:** Emphasize that this is a **workshop proposal** meant to spur collaboration and exploration, not a finalized solution.

---

## ✅ **Final Fairness Judgment**
For a **4-page workshop submission**:
- **dsDV** and **DT7u** provided fair, constructive reviews.
- **PSoS** and **tohC** were overly harsh and missed the workshop’s spirit of idea-sharing and early-stage work.

The paper would likely benefit from a **short revision** that:
- Clarifies notation upfront
- Adds 1–2 sentences explicitly linking to world modeling use cases
- Possibly includes a small extra experiment (e.g., testing a small neural net as surrogate) if space allows

**Bottom line:** The split in reviews is common for workshop submissions. The positive and constructive-negative reviews suggest the paper is **workshop-appropriate** and could be accepted with a clear rebuttal and minor revisions.

Let me know if you'd like help drafting a rebuttal or revising the submission for resubmission.
```

# Qwen 

```markdown
# Analysis of Review Fairness for DISTROSIMULATOR Paper

After examining all four reviews against the paper content and the World Modeling Workshop's stated themes, I find significant inconsistencies in review fairness:

## Review 1 (Reviewer DT7u - Weak Reject)
This review is largely **fair and constructive**. The reviewer:
- Acknowledged the innovative approach and technical rigor
- Provided specific, substantive criticisms about:
  - Limited demonstration of the framework's generality
  - Unclear contribution of each component
  - Need for better contextualization within generative modeling literature
  - Weak demonstration of world modeling applications
  
The "Weak Reject" rating is reasonable given their concerns about the paper's connection to world modeling specifically, though their assessment shows they engaged deeply with the technical content.

## Review 2 (Reviewer PSoS - Strong Reject)
This review demonstrates **significant unfairness** due to fundamental misunderstandings:
- The reviewer incorrectly claims that noise X and observation Y are independent variables, stating "E[y|x] = E[y]" - this misunderstands the core methodology
- The entire framework is designed to create a dependency between X and Y through learning
- Calling the approach "trivial" based on this misunderstanding is unjustified
- The dismissal of relevance to world modeling ignores Section 4.2 where the authors explicitly discuss applications to "generative transition models, causal intervention modeling, physically plausible dynamics, and interpretable scientific discovery"
- The paper directly addresses workshop topics including "Generative world modeling," "Model-based RL and planning," and "Physical plausibility of world models"

This review fails to properly evaluate the paper on its merits and contains technical errors in understanding the method.

## Review 3 (Reviewer dsDV - Accept)
This is a **fair, balanced assessment** that:
- Recognized the computational efficiency and stability advantages
- Appreciated the statistically grounded approach
- Acknowledged current limitations in scalability and temporal modeling
- Properly evaluated the connection to world modeling topics
- Provided constructive feedback despite recommending acceptance

The reviewer correctly identified how this approach addresses core workshop topics related to efficient, stable world modeling.

## Review 4 (Reviewer tohC - Strong Reject)
This review is **partially fair but disproportionately harsh**:
- Valid criticisms about presentation issues and clarity
- Reasonable concerns about dataset complexity and high-dimensional scaling
- However, the "Poor" topic fit rating ignores the paper's explicit discussion of world modeling applications
- The "Strong Reject" recommendation is excessive given the valid conceptual contribution
- The review doesn't adequately acknowledge that workshop submissions (as stated on the website) can include "early-stage results" and the authors positioned this as preliminary work establishing a framework

## Overall Assessment
The reviews show significant disparity in understanding and fairness. Review 2 is particularly problematic as it rejects the paper based on a fundamental misunderstanding of the method. The workshop explicitly welcomes "early-stage results" and "novel datasets," making some criticisms about comprehensive benchmarking and high-dimensional performance less relevant to the workshop context. The paper clearly positions its relevance to world modeling through generative transition models and model-based RL, which some reviewers overlooked.

A fair evaluation would recognize this as a promising preliminary framework with demonstrated computational advantages, while acknowledging limitations that could be addressed in future work - aligning most closely with Reviewer dsDV's assessment.
```

# Mistral 

```markdown
To assess the fairness of the reviews for the DISTROSIMULATOR paper, let's break down the key points from each review and evaluate their objectivity, relevance, and constructive feedback:

---

### **1. Reviewer DT7u**
**Fairness Assessment:**
- **Strengths:** The reviewer acknowledges the innovation and technical rigor of the method, appreciating the mathematical formalism and the general framework.
- **Weaknesses:** The reviewer raises valid concerns about the lack of demonstration of the framework's generality (e.g., not trying other supervised methods) and the unclear contribution of each component (supervised module vs. sampling module). The reviewer also questions the connection to GANs and suggests contextualizing the work within broader generative modeling literature.
- **Questions:** The reviewer asks for clarification on the necessity of each component, the effects of pairing random samples, and the use of PCA for images. These are constructive and relevant questions.
- **Rating:** "Weak Reject" with a confidence of 3/5.

**Fairness:** The review is **fair**. The reviewer provides both praise and constructive criticism, focusing on the paper's limitations and areas for improvement. The rating align't perfectly with the positive aspects, but the feedback is balanced.

---

### **2. Reviewer PSoS**
**Fairness Assessment:**
- **Strengths:** The reviewer summarizes the method accurately.
- **Weaknesses:** The reviewer claims the method is trivial and not novel, arguing that the supervised component reduces to learning a constant function (the mean of the data). This is a strong claim and could be seen as dismissive without empirical evidence from the authors.
- **Criticism:** The reviewer states the experiments are trivial and unrelated to world modeling, and the connection to world models is weak.
- **Rating:** "Strong Reject" with a confidence of 4/5.

**Fairness:** The review is **harsh and potentially unfair**. The reviewer makes a strong technical claim (that the method reduces to learning a constant) without providing evidence or considering the authors' empirical results. The dismissal of the experiments and topic fit is subjective and not sufficiently justified.

---

### **3. Reviewer dsDV**
**Fairness Assessment:**
- **Strengths:** The reviewer praises the framework's computational efficiency, stability, and accessibility, highlighting its potential to challenge compute-intensive architectures.
- **Weaknesses:** The reviewer acknowledges limitations in scalability and temporal fidelity but frames them as areas for future work rather than flaws.
- **Rating:** "Accept" with a confidence of 4/5.

**Fairness:** The review is **fair and positive**. The reviewer balances praise with constructive criticism, focusing on the paper's strengths and potential impact.

---

### **4. Reviewer tohC**
**Fairness Assessment:**
- **Strengths:** The reviewer finds the idea of distribution matching interesting.
- **Weaknesses:** The reviewer criticizes the paper's clarity, flow, and lack of evidence for its usefulness compared to existing methods. The reviewer also questions the relevance to the workshop.
- **Rating:** "Strong Reject" with a confidence of 3/5.

**Fairness:** The review is **partially fair**. The criticism about clarity and relevance is valid, but the reviewer does not engage deeply with the technical content or empirical results. The rating seems overly negative given the acknowledged interest in the idea.

---

## **Overall Fairness Assessment**

| Reviewer | Fairness | Constructiveness | Objectivity | Rating |
|----------|----------|------------------|-------------|--------|
| DT7u     | Fair     | High             | High        | 2      |
| PSoS     | Harsh    | Low              | Low         | 1      |
| dsDV     | Fair     | High             | High        | 4      |
| tohC     | Partial  | Medium           | Medium      | 1      |

### **Key Observations:**
- **DT7u and dsDV** provide balanced, constructive feedback and are fair in their assessments.
- **PSoS** is overly dismissive and makes strong claims without sufficient evidence, which is unfair.
- **tohC** raises valid points about clarity and relevance but does not engage deeply with the technical content, leading to a harsh rating.

### **Conclusion:**
The reviews by **DT7u** and **dsDV** are fair and constructive. The reviews by **PSoS** and **tohC** are less fair, with **PSoS** being particularly harsh and dismissive. The authors should address the technical concerns raised by **DT7u** and **PSoS** (e.g., demonstrating the necessity of each component, exploring other supervised methods) and improve clarity as suggested by **tohC**. The positive feedback from **dsDV** should be highlighted in any revisions.

Would you like a suggested response to the reviewers or a strategy for revising the paper?
```

# Gemini

```markdown
This is a detailed assessment of the fairness of the four official reviews for the DISTROSIMULATOR paper.

Overall, the reviews exhibit a **high degree of contradiction**, particularly regarding **Topic Fit** and **Soundness**, which raises concerns about the uniformity and fairness of the evaluation. Reviewers PSoS and tohC recommend a Strong Reject, Reviewer DT7u recommends a Weak Reject, and Reviewer dsDV recommends an Accept.

---

## 🧐 Review Assessment

### 1. Topic Fit (Contradictory)
There is a fundamental disagreement on the paper's relevance to the workshop.

| Reviewer | Topic Fit Rating | Comment | Fairness Assessment |
| :---: | :---: | :---: | :---: |
| **dsDV** | 3 (Good) | "Directly addresses the workshop’s core topics." | Fair |
| **DT7u** | 2 (Fair) | "Somewhat related to the workshop’s themes." | Fair |
| **PSoS** | 1 (Poor) | "Not related to the workshop’s themes." | **Unfair (Contradictory)** |
| **tohC** | 1 (Poor) | "Unclear how the proposed approach is relevant to the workshop." | **Unfair (Contradictory)** |

* **Conclusion:** The workshop topics include "Generative world modeling" and "Model-based RL and planning." [cite_start]The paper explicitly discusses the utility for **world modeling tasks** like "generative transition models and model-based RL" [cite: 44] [cite_start]and "model environment dynamics and transitions" [cite: 68][cite_start], and mentions the advantages for "generative transition models, causal intervention modeling, physically plausible dynamics"[cite: 315]. Reviewers PSoS and tohC's "Poor" rating, which forms a basis for their rejection, seems **unfair** given the paper's explicit positioning within these relevant domains.

---

### 2. Soundness and Methodology (Highly Contradictory)
The most critical contradiction lies in the core mathematical interpretation.

* **Reviewer dsDV:** Rates Soundness **3 (Good)**, stating, "Math is correct and all claims are justified."
* **Reviewer PSoS:** Rates Soundness **1 (Poor - Major errors)**. Their primary argument is a major flaw: since $X$ (noise) and $Y$ (data) are sampled independently, the Bayes optimal predictor $f^*(X)$ should be the constant $\mathbb{E}[Y]$. This would render the method trivial, as it reduces to fitting residuals around the mean, which is not novel. **This is a strong, technical critique that directly questions the method's soundness.**
    * [cite_start]*Self-Correction:* The paper specifies the problem as learning a generator $G: \mathbb{R}^d \rightarrow \mathbb{R}^d$ such that $G(X) \sim P_Y$, where $X \sim \mathcal{N}(0, I_d)$[cite: 117]. [cite_start]The training step uses $X$ and $Y$ in a supervised regression $\hat{f}_{\theta}(X) \approx \mathbb{E}[Y|X]$[cite: 118]. **Reviewer PSoS is technically correct under a standard IID regression setup where $X$ and $Y$ are independent.** The model is trained on *randomly paired* IID noise $X$ and data $Y$, which, mathematically, means $\mathbb{E}[Y|X] = \mathbb{E}[Y]$ if $X$ and $Y$ are truly independent features/targets. The authors need to address this apparent contradiction, as it invalidates the central $\mathbb{E}[Y|X]$ decomposition if the surrogate model is truly performing an optimal MSE fit. This makes PSoS's critique **fair and highly relevant** to the paper's core claim.
* **Reviewer DT7u:** Rates Soundness **2 (Fair - Small errors)**.
* **Reviewer tohC:** Rates Soundness **2 (Fair - Small errors)**.

* **Conclusion:** The contradictory Soundness ratings indicate a severe lack of consensus on the method's fundamental mathematical validity. Reviewer PSoS's argument is well-reasoned and directly challenges the premise of the training objective (Eq. 3) given the independent sampling of $X$ and $Y$. The fact that Reviewer dsDV rates it "Good" without acknowledging this issue suggests a potential oversight. The paper needs to either refute PSoS's claim or explain the non-Bayesian behavior of the KRR surrogate in this specific, highly-regularized context.

---

### 3. Contribution and Weaknesses (Fairly Consistent)
There is a consensus on the need for more evidence.

* [cite_start]**Generality:** Reviewers DT7u, PSoS, and tohC all criticize the lack of demonstration of **model-agnostic generality** by only using Kernel Ridge Regression (KRR)[cite: 105, 143]. [cite_start]This is a fair point given the framework's main selling point[cite: 45].
* **Ablation/Criticality of Components:** Reviewer DT7u and PSoS both request an **ablation study** to determine the necessity of the residual sampling component versus the surrogate model, questioning which part does the "most legwork." This is a standard and fair request for validating a decomposed framework.
* **Relative Benefit/Benchmarking:** Reviewer DT7u questions the lack of demonstration of the purported benefits (computational cost, lack of mode collapse) against baselines like GANs. [cite_start]Reviewer dsDV praises the efficiency but also notes the lack of direct benchmarks[cite: 309]. [cite_start]This is acknowledged as a limitation by the authors for the preliminary work [cite: 309] but is a fair critique from the reviewers for a conference submission.
* [cite_start]**High-Dimensional Data:** Reviewer DT7u and tohC correctly point out the reliance on **PCA for image data**[cite: 213, 306], limiting the demonstrated capability for high-dimensional raw data. [cite_start]This is an explicit limitation in the paper[cite: 307].

---

## ⚖️ Final Verdict

The reviews are **unbalanced and contradictory**.

* **Reviewer PSoS's and Reviewer tohC's "Strong Reject" ratings are questionable,** as they rely heavily on a "Poor" Topic Fit rating that is contradicted by another reviewer and the paper's text.
* **Reviewer PSoS's mathematical critique is the most substantive and potentially damning.** If $\mathbb{E}[Y|X] \approx \mathbb{E}[Y]$ holds for the trained model, the core generative mechanism is trivial, making the novelty and contribution minimal (as PSoS suggests). This point must be addressed to ensure fairness.
* **Reviewer dsDV's "Accept" rating** seems optimistic and fails to engage with the mathematical soundness issue and the key requests for ablation and generality.
* **Reviewer DT7u's "Weak Reject"** is the most balanced, highlighting the innovative idea while focusing on standard technical requirements: lack of generality demonstration, no ablation study, and missing comparative benchmarks on the claimed benefits.

The paper would benefit greatly from addressing **Reviewer PSoS's mathematical critique** and providing the **ablation study** requested by DT7u and PSoS.

Would you like me to draft a response to the reviewers focusing on the core contradictions and requested experiments?
```

# Claude

```markdown
Thank you for this crucial context! This **significantly changes** my assessment. Knowing this is a **4-page workshop submission** (not a full conference paper) makes several reviews substantially unfair.

## Revised Fairness Assessment

### **Context Matters: Workshop vs. Conference Standards**

Workshops are explicitly for:
- ✓ Early-stage results
- ✓ Novel methodological ideas
- ✓ Opinion pieces
- ✓ Preliminary explorations

The paper clearly states: *"This paper presents preliminary work exploring a novel model-agnostic framework"* and *"detailed benchmarking against specific baselines left for future work."*

---

## Review-by-Review Analysis (REVISED)

### **Reviewer PSoS - "Strong Reject" (1/5)** ❌ **UNFAIR**

**Multiple fairness violations:**

1. **Technical misunderstanding** (as noted before): Claims the method is trivial because f*(x) = E[y], missing that the framework intentionally pairs noise with data to learn a generative transformation

2. **Inappropriate standards**: "The experiments are trivial" - but this is a **4-page workshop paper** presenting preliminary work. MNIST/Fashion-MNIST are reasonable for demonstrating feasibility.

3. **Dismissive tone**: "quite frankly, trivial" is unprofessional for any review, especially for exploratory workshop submissions

**Verdict**: **Highly unfair**. Applies conference-level expectations to workshop submission and fundamentally misunderstands the method.

---

### **Reviewer DT7u - "Weak Reject" (2/5)** ⚠️ **MOSTLY FAIR, but slightly harsh**

**Good aspects:**
- Constructive, professional tone
- Specific, actionable questions
- Balanced critique

**Workshop context issues:**
- Criticizes lack of demonstrations with other supervised methods, but the authors explicitly frame this as preliminary work and invite community exploration
- Requests ablation studies that may be beyond 4-page scope
- The paper explicitly acknowledges: "detailed benchmarking against specific baselines left for future work"

**However**: The criticism about not demonstrating claimed advantages (computational cost, stability) is valid even for a workshop paper - if you claim something, show it.

**Verdict**: **Mostly fair**, though could be more lenient given workshop context. The weaknesses identified are real, but the scoring might be harsh for preliminary work.

---

### **Reviewer dsDV - "Accept" (4/5)** ⚠️ **SUSPICIOUS**

My previous concerns remain:
- Uses the paper's own terminology uncritically
- Overly enthusiastic tone seems promotional
- Acknowledges weaknesses but dismisses them too easily

**However**, in workshop context:
- Recognizing novelty and potential is appropriate
- "Early-stage" frameworks deserve encouragement

**Verdict**: Still **potentially biased**, but acceptance is more reasonable for a workshop exploring novel ideas.

---

### **Reviewer tohC - "Strong Reject" (1/5)** ❌ **UNFAIR**

**Major fairness issues for workshop context:**

1. **"Very simplistic datasets (MNIST)"** - This is standard for proof-of-concept in 4-page papers! The authors use 7 different datasets across different domains.

2. **"I don't think the paper provides enough evidence... better than neural network-based approaches"** - The paper **explicitly** says this is preliminary and invites the community to explore neural surrogates.

3. **Topic fit criticism** - While debatable, the paper does discuss "generative world modeling," "generative transition models," and "model-based RL" which are listed topics.

4. **Nitpicking**: Criticizing "title is all caps" is inappropriate for a substantive review

**Verdict**: **Unfair**. Applies unreasonable standards for a 4-page workshop paper explicitly presenting early-stage ideas.

---

## Key Fairness Issues

### **1. Mismatched Expectations** 
Three reviewers (PSoS, tohC, and partially DT7u) seem to evaluate this as a full conference paper rather than a **4-page workshop extended abstract** meant for early-stage ideas.

### **2. The Paper Explicitly Addresses Their Criticisms**

The authors state:
- *"preliminary work exploring a novel model-agnostic framework"*
- *"detailed benchmarking against specific baselines left for future work"*
- *"We invite workshop participants to try different surrogates"*
- *"Open question for the community: Which surrogates work best?"*

This is **exactly what workshops are for** - presenting novel ideas and inviting community exploration.

### **3. Valid Criticisms (Even for Workshop)**

- **Weak world modeling connection**: Only briefly mentioned in introduction and discussion
- **Missing ablations**: Should show that both surrogate + residual sampling are necessary (this could fit in 4 pages)
- **No timing comparisons**: If claiming computational efficiency, compare to at least one GAN baseline
- **Overclaiming**: Statements like "decades of supervised learning research become immediately applicable" need tempering

---

## Revised Recommendations

### **For Program Chairs:**

1. **Disregard or discount** Reviewer PSoS (technical misunderstanding + wrong standards)
2. **Disregard or discount** Reviewer tohC (unreasonable workshop standards)
3. **Investigate** Reviewer dsDV for potential bias/conflict
4. **Weight heavily** Reviewer DT7u's feedback (most fair and constructive)

### **For Authors (if revising):**

**Critical additions** (even in 4 pages):
1. **Simple ablation**: Show performance with just surrogate (no residual sampling) vs. full method
2. **One timing comparison**: Train a simple GAN on Fashion-MNIST, compare training time
3. **Strengthen world modeling connection**: Add 1-2 sentences on how this specifically helps transition models or model-based RL with concrete examples
4. **Temper claims**: Change "immediately applicable" to "potentially applicable with further research"

**Optional for workshop context:**
- Testing other surrogates (explicitly left for future work is fine)
- High-dimensional images without PCA (acknowledged limitation is fine)
- Extensive benchmarking (inappropriate for 4-page workshop)

---

## Final Verdict: **Review Process is UNFAIR**

**Score distribution:** 1 - 2 - 4 - 1 (average: 2.0)

**Fair assessment for 4-page workshop:** Should be **2.5-3.0** (Weak Accept to Borderline)

**Rationale:**
- Novel, interesting idea appropriate for workshop discussion ✓
- Demonstrates feasibility across multiple domains ✓
- Computational efficiency shown (though not compared) ✓
- Explicitly acknowledges preliminary nature ✓
- Weak world modeling connection ✗
- Missing simple ablations (feasible in 4 pages) ✗
- Some overclaiming ✗

**Recommendation:** **Weak Accept** with minor revisions. This is exactly the kind of early-stage, novel methodological exploration workshops should encourage, despite its limitations.
```
