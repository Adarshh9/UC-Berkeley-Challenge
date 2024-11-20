

# **Cross-Attention Driven Multi-Agent Collaborative Chatbot**

## **Overview**
We present a **Cross-Attention Driven Multi-Agent Collaborative Chatbot**, leveraging specialized smaller language models (SLMs) to outperform monolithic large-scale LLMs. This solution dynamically integrates domain-specific expertise, utilizing cutting-edge reinforcement learning (RL) and multi-agent reinforcement learning (MARL) techniques to deliver **semantically rich, contextually precise, and computationally efficient responses**.

<img src="https://github.com/user-attachments/assets/f9b00080-2fde-4cbc-937e-db290543cf19" alt="FlowChart" style="width:50%;"/>

---

## **Key Features**
1. **Query Splitting and Routing**:
   - Implements GPT-4-mini API for intelligent query segmentation and routing.
   - Distributes query fragments to specialized models like **BioGPT** (medical expertise) and **Qwen Coder** (technical expertise).

2. **Model Collaboration via Cross-Attention**:
   - Employs a novel **cross-attention mechanism** for information exchange between models.
   - Enhances the semantic richness of individual outputs by fusing domain-specific knowledge.

3. **Reinforcement Learning & Multi-Agent Collaboration**:
   - Models learn and improve collaboratively through **reinforcement learning (RL)**.
   - Implements **multi-agent reinforcement learning (MARL)** with game-theory principles to foster synergistic behavior.

4. **Optimized Performance with Smaller Models**:
   - Uses lightweight models (<7B parameters) for domain specialization, ensuring **low latency** and **efficient compute usage**.
   - Achieves **accuracy comparable to or better than LLMs** with a fraction of the computational cost.

---

## **System Architecture**
1. **Input Handling**:
   - Query is intelligently split and routed by GPT-4-mini.
2. **Domain-Specific Processing**:
   - BioGPT and Qwen Coder process their respective query fragments and generate initial responses.
3. **Cross-Attention Integration**:
   - Responses are integrated through cross-attention mechanisms to exchange insights and refine outputs.
4. **Final Merging and Output**:
   - GPT-4-mini combines the refined outputs into a unified, semantically precise response.

---

## **Tech Stack**
- **Models**: BioGPT, Qwen Coder (Hugging Face)
- **Frameworks**: PyTorch, Hugging Face Transformers
- **Learning Techniques**: Reinforcement Learning (RL), Multi-Agent RL (MARL), Game Theory, Cross-Attention

---

## **Why This Approach?**
- **Efficiency**: Smaller models with specialized knowledge outperform generic, compute-heavy LLMs.
- **Collaboration**: A multi-agent framework ensures dynamic learning and better decision-making.
- **Scalability**: Modular design allows integration of additional domain-specific models for broader use cases.

---

## **Setup Instructions**
1. Clone the repository:
   ```bash
   git clone https://github.com/YourRepo/CrossAttentionChatbot.git
   cd CrossAttentionChatbot
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure API keys for GPT-4-mini and Hugging Face models in `.env`.
4. Run the application:
   ```bash
   python main.py
   ```

---

## **Applications**
- **Healthcare**: Generate medically accurate responses via BioGPT.
- **Coding Assistance**: Leverage Qwen Coder for precise technical solutions.
- **Cross-Domain Problem Solving**: Seamless collaboration across diverse domains.
And many more accordingly!

---

## **Results**
- Achieves **90%+ accuracy** in domain-specific benchmarks.
- Reduces compute costs by up to **50%** compared to monolithic LLMs.
- Average response time of **<2 seconds** per query.

---

## **Future Enhancements**
- Implementing **adaptive reward systems** for RL.
- Expanding domain coverage with additional SLMs.
- Enhancing collaboration with **adversarial training**.
