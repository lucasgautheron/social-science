import pandas as pd
import numpy as np
from scipy.special import gammaln, beta
from scipy.optimize import minimize_scalar
from scipy.stats import pareto, beta as beta_dist
from typing import List, Optional, Set, Tuple, NamedTuple
from dataclasses import dataclass
import random
import pickle
import re

# JAX imports
import jax.numpy as jnp
import jax
from jax import grad, jit, vmap
from jax.scipy.special import gammaln as jax_gammaln
import jax.scipy.optimize as jopt

# Enable 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)

topic_labels = pd.read_csv("data/topic_list.csv").set_index("Topic")["Name"].to_dict()

def query_summary(topics: List[str], model: str = "gpt-3.5-turbo") -> str:
    """
    Query one comprehensive summary for all topics using the newer OpenAI Python client.

    Args:
        topics (List[str]): List of topics to summarize together
        api_key (str): Your OpenAI API key
        model (str): The model to use (default: gpt-3.5-turbo)

    Returns:
        str: A single summary covering all topics
    """
    from openai import OpenAI

    api_key = "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"

    client = OpenAI(api_key=api_key)

    try:
        # Create a prompt that asks for one summary of all topics
        topics_str = ", ".join(topics)
        prompt = f"Provide a short, 3-4 word label summarizing the branch occupied by these topics within social science: {topics_str}."

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that provides comprehensive, well-structured summaries that connect related concepts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=30,  # Increased for comprehensive summary
            temperature=0.7
        )

        summary = response.choices[0].message.content.strip()
        return summary

    except Exception as e:
        return f"Error: Could not retrieve summary - {str(e)}"


@dataclass
class TreeNode:
    """Represents a node in a binary Dirichlet tree."""
    node_id: int
    children: List['TreeNode']
    parent: Optional['TreeNode'] = None
    is_leaf: bool = False
    leaf_indices: Set[int] = None
    alpha: Optional[float] = None  # Alpha parameter for beta distribution
    beta_param: Optional[float] = None  # Beta parameter for beta distribution
    p_values: Optional[np.ndarray] = None  # Estimated p[i] for each valid sample
    valid_sample_indices: Optional[np.ndarray] = None  # Indices of valid samples
    counts: np.ndarray = None

    def __post_init__(self):
        if self.leaf_indices is None:
            self.leaf_indices = set()


class OptimizationState(NamedTuple):
    """State for JAX optimization"""
    scale: float
    pi: float
    p_values: jnp.ndarray


class JAXDirichletOptimizer:
    """
    JAX-optimized version of Dirichlet tree parameter learning.
    Provides significant speedup through JIT compilation and auto-differentiation.
    """

    def __init__(self):
        # JIT compile only the core computational functions
        self.log_likelihood_jit = jit(self._log_likelihood_node)
        self.log_posterior_jit = jit(self._log_posterior)

        self.min_scale = 0.25
        self.max_scale = 100.0

    @staticmethod
    def _log_pareto_pdf(x: float, xm: float = 1, alpha: float = 1.5) -> float:
        """Log PDF of Pareto distribution - JAX compatible"""
        return jnp.where(
            x >= xm,
            jnp.log(alpha) + alpha * jnp.log(xm) - (alpha + 1) * jnp.log(x),
            -jnp.inf
        )

    @staticmethod
    def _log_beta_pdf(x: float, alpha: float, beta: float) -> float:
        """Log PDF of Beta distribution - JAX compatible"""
        return jnp.where(
            (x > 0) & (x < 1),
            (alpha - 1) * jnp.log(x) + (beta - 1) * jnp.log(1 - x) -
            jax_gammaln(alpha) - jax_gammaln(beta) + jax_gammaln(alpha + beta),
            -jnp.inf
        )

    @staticmethod
    def _log_likelihood_node(counts: jnp.ndarray, p_values: jnp.ndarray,
                             alpha: float, beta_param: float) -> float:
        """
        Vectorized log likelihood computation - JAX optimized.
        All operations are JIT compiled for maximum performance.
        """
        # Extract counts
        left_counts = counts[:, 0]
        right_counts = counts[:, 1]
        total_counts = left_counts + right_counts

        # Filter valid samples
        valid_mask = total_counts > 0
        valid_counts_left = jnp.where(valid_mask, left_counts, 0)
        valid_counts_right = jnp.where(valid_mask, right_counts, 0)
        valid_counts_total = jnp.where(valid_mask, total_counts, 1)  # Avoid log(0)
        valid_p = jnp.where(valid_mask, p_values, 0.5)  # Safe default

        # Check bounds - return -inf if any p[i] out of bounds
        p_in_bounds = jnp.all((valid_p > 0) & (valid_p < 1))

        # Multinomial coefficients (vectorized)
        multinomial_coeff = (
                jax_gammaln(valid_counts_total + 1) -
                jax_gammaln(valid_counts_left + 1) -
                jax_gammaln(valid_counts_right + 1)
        )

        # Multinomial probabilities (vectorized)
        multinomial_prob = (
                valid_counts_left * jnp.log(valid_p) +
                valid_counts_right * jnp.log(1 - valid_p)
        )

        # Beta priors (vectorized)
        beta_prior = (
                (alpha - 1) * jnp.log(valid_p) +
                (beta_param - 1) * jnp.log(1 - valid_p) -
                jax_gammaln(alpha) - jax_gammaln(beta_param) + jax_gammaln(alpha + beta_param)
        )

        # Sum only over valid samples
        sample_log_likelihoods = multinomial_coeff + multinomial_prob + beta_prior
        total_likelihood = jnp.sum(jnp.where(valid_mask, sample_log_likelihoods, 0.0))

        return jnp.where(p_in_bounds, total_likelihood, -jnp.inf)

    def _log_posterior(self, state: OptimizationState, counts: jnp.ndarray) -> float:
        """Compute log posterior for given state"""
        scale, pi, p_values = state.scale, state.pi, state.p_values

        # Bounds checking
        scale_valid = scale > self.min_scale
        pi_valid = (pi > 0) & (pi < 1)

        alpha = scale * pi
        beta_param = scale * (1 - pi)

        # Log likelihood
        log_likelihood = self._log_likelihood_node(counts, p_values, alpha, beta_param)

        # Hyperparameter priors
        log_prior_scale = self._log_pareto_pdf(scale, self.min_scale, 1.0)
        log_prior_pi = self._log_beta_pdf(pi, 1.0, 1.0)

        log_posterior = log_likelihood + log_prior_scale + log_prior_pi

        return jnp.where(scale_valid & pi_valid, log_posterior, -jnp.inf)

    def _optimize_hyperparams_step(self, state: OptimizationState, counts: jnp.ndarray) -> OptimizationState:
        """Optimize hyperparameters with fixed p_values using JAX optimization"""

        def objective(hyperparams):
            scale, pi = hyperparams
            new_state = OptimizationState(scale, pi, state.p_values)
            return -self._log_posterior(new_state, counts)

        # Use JAX optimizers instead of manual gradient descent
        from jax.example_libraries import optimizers

        # Initialize optimizer
        opt_init, opt_update, get_params = optimizers.adam(step_size=0.01)

        initial_hyperparams = jnp.array([state.scale, state.pi])
        opt_state = opt_init(initial_hyperparams)

        # Fixed number of optimization steps
        def update_step(i, opt_state):
            params = get_params(opt_state)
            # Project to bounds
            params = jnp.array([
                jnp.clip(params[0], self.min_scale + 0.01, self.max_scale),  # scale
                jnp.clip(params[1], 0.001, 0.999)  # pi
            ])
            grads = grad(objective)(params)
            return opt_update(i, grads, opt_state)

        # Run optimization loop
        for i in range(100):
            opt_state = update_step(i, opt_state)

        final_params = get_params(opt_state)
        # Final projection to bounds
        final_params = jnp.array([
            jnp.clip(final_params[0], self.min_scale + 0.01, self.max_scale),
            jnp.clip(final_params[1], 0.001, 0.999)
        ])

        return OptimizationState(final_params[0], final_params[1], state.p_values)

    def _optimize_p_values_step(self, state: OptimizationState, counts: jnp.ndarray) -> OptimizationState:
        """Optimize p_values with fixed hyperparameters using JAX optimization"""

        alpha = state.scale * state.pi
        beta_param = state.scale * (1 - state.pi)

        def objective(p_vals):
            return -self._log_likelihood_node(counts, p_vals, alpha, beta_param)

        # Use JAX optimizers
        from jax.example_libraries import optimizers

        # Initialize optimizer
        opt_init, opt_update, get_params = optimizers.adam(step_size=0.1)
        opt_state = opt_init(state.p_values)

        # Fixed number of optimization steps
        def update_step(i, opt_state):
            params = get_params(opt_state)
            # Project to bounds (0, 1)
            params = jnp.clip(params, 0.001, 0.999)
            grads = grad(objective)(params)
            return opt_update(i, grads, opt_state)

        # Run optimization loop
        for i in range(200):
            opt_state = update_step(i, opt_state)

        final_p_values = get_params(opt_state)
        # Final projection to bounds
        final_p_values = jnp.clip(final_p_values, 0.001, 0.999)

        return OptimizationState(state.scale, state.pi, final_p_values)

    def optimize_node_parameters(self, counts: np.ndarray, max_iterations: int = 10) -> Tuple[float, float, np.ndarray]:
        """
        Main optimization function using JAX.

        Args:
            counts: numpy array of shape (n_samples, 2) with [left_count, right_count]
            max_iterations: maximum number of alternating optimization iterations

        Returns:
            Tuple of (alpha, beta, p_values)
        """
        # Convert to JAX arrays
        counts_jax = jnp.array(counts)
        n_samples = counts.shape[0]

        # Initialize
        total_counts = counts[:, 0] + counts[:, 1]
        p_init = (counts[:, 0] + 0.5) / (total_counts + 1.0)
        p_init_jax = jnp.array(p_init)

        # Initial state
        state = OptimizationState(
            scale=2.0,
            pi=0.5,
            p_values=p_init_jax
        )

        print(f"JAX optimization: {n_samples} samples")
        print(f"Initial p[i] range: [{float(jnp.min(p_init_jax)):.3f}, {float(jnp.max(p_init_jax)):.3f}]")

        prev_log_posterior = -jnp.inf

        for iteration in range(max_iterations):
            # Step 1: Optimize hyperparameters
            state = self._optimize_hyperparams_step(state, counts_jax)

            # Step 2: Optimize p_values
            state = self._optimize_p_values_step(state, counts_jax)

            # Check convergence
            current_log_posterior = self._log_posterior(state, counts_jax)

            if iteration > 0:
                improvement = float(current_log_posterior - prev_log_posterior)
                print(
                    f"  Iteration {iteration}: log_posterior = {float(current_log_posterior):.2f}, improvement = {improvement:.4f}")

                if abs(improvement) < 1e-6:
                    print(f"  JAX converged after {iteration + 1} iterations")
                    break
            else:
                print(f"  Iteration {iteration}: log_posterior = {float(current_log_posterior):.2f}")

            prev_log_posterior = current_log_posterior

        # Extract final results
        final_alpha = float(state.scale * state.pi)
        final_beta = float(state.scale * (1 - state.pi))
        final_p_values = np.array(state.p_values)

        print(f"  Final hyperparameters: α={final_alpha:.3f}, β={final_beta:.3f}")
        print(f"  Final p[i] range: [{final_p_values.min():.3f}, {final_p_values.max():.3f}]")

        return final_alpha, final_beta, final_p_values


class SimplifiedDirichletTreeLearner:
    """
    Simplified learner for binary Dirichlet trees with known structure.
    Each internal node has exactly 2 children and uses a beta distribution.
    Learns alpha and beta parameters using the prior:
    (alpha, beta) = scale * (p, 1-p) where scale ~ Pareto(1, 1.5) and p ~ Beta(1, 1)
    """

    def __init__(self):
        self.next_node_id = 0
        self.jax_optimizer = JAXDirichletOptimizer()

    def _get_next_node_id(self) -> int:
        """Get next unique node ID."""
        node_id = self.next_node_id
        self.next_node_id += 1
        return node_id

    def _create_leaf_node(self, leaf_idx: int) -> TreeNode:
        """Create a leaf node for a specific category."""
        node = TreeNode(
            node_id=self._get_next_node_id(),
            children=[],
            is_leaf=True,
            leaf_indices={leaf_idx}
        )
        return node

    def _create_internal_node(self, children: List[TreeNode]) -> TreeNode:
        """Create an internal node with exactly 2 children."""
        if len(children) != 2:
            raise ValueError("Binary tree nodes must have exactly 2 children")

        node = TreeNode(
            node_id=self._get_next_node_id(),
            children=children,
            is_leaf=False
        )

        # Set parent relationships and collect leaf indices
        leaf_indices = set()
        for child in children:
            child.parent = node
            leaf_indices.update(child.leaf_indices)
        node.leaf_indices = leaf_indices

        return node

    def _compute_node_counts(self, node: TreeNode, data: np.ndarray) -> np.ndarray:
        """
        Compute count data for a binary node.
        Returns counts for left and right children.
        """
        if node.is_leaf:
            return data[:, list(node.leaf_indices)[0]]

        # For internal nodes, sum counts for each child's subtree (binary only)
        left_child = node.children[0]
        right_child = node.children[1]

        left_counts = np.sum(data[:, list(left_child.leaf_indices)], axis=1)
        right_counts = np.sum(data[:, list(right_child.leaf_indices)], axis=1)

        return np.column_stack([left_counts, right_counts])  # Shape: (n_samples, 2)

    # def _log_pareto_pdf(self, x, xm=1.0, alpha=1.5):
    #     """Log PDF of Pareto distribution."""
    #     if x < xm:
    #         return -np.inf
    #     return np.log(alpha) + alpha * np.log(xm) - (alpha + 1) * np.log(x)

    def _log_beta_pdf(self, x, alpha, beta_param):
        """Log PDF of Beta distribution."""
        if x <= 0 or x >= 1:
            return -np.inf
        return (alpha - 1) * np.log(x) + (beta_param - 1) * np.log(1 - x) - np.log(beta(alpha, beta_param))

    def _log_likelihood_node(self, counts: np.ndarray, p_values: np.ndarray, alpha: float, beta_param: float) -> float:
        """
        Compute log likelihood for hierarchical model WITHOUT integrating out p[i].

        GENERATIVE MODEL:
        For each sample i:
        1. p[i] ~ Beta(alpha, beta)  # Sample-specific probability (TO BE ESTIMATED)
        2. counts[i] ~ Multinomial(total_count[i], [p[i], 1-p[i]])  # Counts given p[i]

        LIKELIHOOD BREAKDOWN (NO INTEGRATION):
        P(counts, p | alpha, beta) = ∏_i P(counts[i] | p[i]) * P(p[i] | alpha, beta)

        Where:
        - P(counts[i] | p[i]) = Multinomial(left_count[i], right_count[i] | total_count[i], p[i])
                              = C(total_count[i], left_count[i]) * p[i]^left_count[i] * (1-p[i])^right_count[i]

        - P(p[i] | alpha, beta) = Beta(p[i] | alpha, beta)
                                = p[i]^(alpha-1) * (1-p[i])^(beta-1) / B(alpha, beta)

        Args:
            counts: Shape (n_samples, 2) - [left_count, right_count] for each sample
            p_values: Shape (n_samples,) - estimated p[i] for each sample
            alpha, beta_param: Beta distribution hyperparameters
        """
        # Extract counts vectorially
        left_counts = counts[:, 0]
        right_counts = counts[:, 1]
        total_counts = left_counts + right_counts

        # Filter out samples with zero total counts
        valid_mask = total_counts > 0
        if not np.any(valid_mask):
            return 0.0

        # Apply mask to all arrays
        left_counts = left_counts[valid_mask]
        right_counts = right_counts[valid_mask]
        total_counts = total_counts[valid_mask]
        p_vals = p_values[valid_mask]

        # Ensure all p[i] are in valid range
        if np.any(p_vals <= 0) or np.any(p_vals >= 1):
            return -np.inf

        # COMPONENT 1: Multinomial likelihood P(counts[i] | p[i]) - VECTORIZED
        # = C(total, left) * p[i]^left * (1-p[i])^right
        multinomial_coeff = (gammaln(total_counts + 1) -
                             gammaln(left_counts + 1) -
                             gammaln(right_counts + 1))

        multinomial_prob = (left_counts * np.log(p_vals) +
                            right_counts * np.log(1 - p_vals))

        # COMPONENT 2: Beta prior P(p[i] | alpha, beta) - VECTORIZED
        # = p[i]^(alpha-1) * (1-p[i])^(beta-1) / B(alpha, beta)
        beta_prior = ((alpha - 1) * np.log(p_vals) +
                      (beta_param - 1) * np.log(1 - p_vals))

        # Subtract log B(alpha, beta) - this is constant for all samples
        log_beta_norm = gammaln(alpha) + gammaln(beta_param) - gammaln(alpha + beta_param)
        beta_prior -= log_beta_norm

        # Total log likelihood - sum over all valid samples
        sample_log_likelihoods = multinomial_coeff + multinomial_prob + beta_prior
        log_likelihood = np.sum(sample_log_likelihoods)

        return log_likelihood

    def _learn_alpha_for_node(self, node: TreeNode, data: np.ndarray) -> tuple:
        """
        Learn alpha, beta, and p[i] parameters for a binary node using MAP estimation.
        Uses efficient alternating optimization with multiple iterations to handle large parameter spaces.

        Uses the prior: (alpha, beta) = scale * (p, 1-p) where scale ~ Pareto(1, 1.5) and p ~ Beta(1, 1)
        """
        if node.is_leaf or len(node.children) != 2:
            return None, None

        for i, child in enumerate(node.children):
            print(f"Child{i}: {[topic_labels[x] for x in sorted(child.leaf_indices)]}")
            # print(f"Child{i}: {query_summary([topic_labels[x] for x in sorted(child.leaf_indices)])}")

        # Get count data for this node
        counts = self._compute_node_counts(node, data)

        # Filter out samples with zero total counts
        valid_samples = counts.sum(axis=1) > 0
        if not np.any(valid_samples):
            # Default to uniform beta if no valid data
            return 1.0, 1.0

        counts = counts[valid_samples]
        n_samples = counts.shape[0]

        # Use JAX optimizer if available and dataset is large enough
        alpha, beta, p_values = self.jax_optimizer.optimize_node_parameters(counts)
        # Store results in node
        node.p_values = p_values
        node.valid_sample_indices = np.where(valid_samples)[0]

        p = counts[:, 0].sum() / counts.sum()
        surprise = -beta_dist.logpdf(p, alpha, beta)
        print(p, surprise)
        #
        # from matplotlib import pyplot as plt
        # plt.hist(p_values[counts.sum(axis=1) > 10], bins=np.linspace(0, 1, 100), density=True)
        # plt.hist((counts[:, 0] / counts.sum(axis=1))[counts.sum(axis=1) > 10], bins=np.linspace(0, 1, 100),
        #          density=True)
        # plt.axvline(p, color="black")
        # plt.title(
        #     f"N={(counts.sum(axis=1) > 10).sum()}, "
        #     f"$\mu$={(counts[:, 0] / counts.sum(axis=1))[counts.sum(axis=1) > 10].mean():.2f}, "
        #     f"$\\alpha/(\\alpha+\\beta)$={alpha / (alpha + beta):.2f}, $-\\log P$={surprise:.2f}"
        # )
        # x = np.linspace(0, 1, 500)
        # y = beta_dist.pdf(x, alpha, beta)
        # plt.plot(x, y)
        # plt.show()

        return alpha, beta, counts

    def _simple_alpha_estimate(self, counts: np.ndarray) -> tuple:
        """Simple method-of-moments estimate for beta parameters as fallback."""
        # Compute sample proportions for left child
        total_counts = counts.sum(axis=1)
        valid_mask = total_counts > 0

        if not np.any(valid_mask):
            return 1.0, 1.0

        proportions = counts[valid_mask, 0] / total_counts[valid_mask]

        # Method of moments for beta distribution
        mean_prop = np.mean(proportions)
        var_prop = np.var(proportions)

        # Avoid division by zero
        if var_prop == 0 or mean_prop == 0 or mean_prop == 1:
            return 1.0, 1.0

        # Method of moments formulas for beta distribution
        common_term = mean_prop * (1 - mean_prop) / var_prop - 1
        alpha = mean_prop * common_term
        beta_param = (1 - mean_prop) * common_term

        # Ensure positive parameters
        alpha = max(alpha, 0.1)
        beta_param = max(beta_param, 0.1)

        return alpha, beta_param

    def learn_tree_parameters(self, root: TreeNode, data: np.ndarray) -> TreeNode:
        """
        Learn alpha and beta parameters for all binary nodes in the tree.
        """
        print("Learning alpha and beta parameters for tree nodes...")

        # Traverse all internal nodes and learn their alphas
        nodes_to_visit = [root]
        nodes_processed = 0

        while nodes_to_visit:
            node = nodes_to_visit.pop(0)

            if not node.is_leaf and len(node.children) == 2:
                print(f"Learning alpha for node {node.node_id} with {len(node.children)} children...")
                node.alpha, node.beta_param, node.counts = self._learn_alpha_for_node(node, data)
                if node.alpha is not None:
                    print(f"  Learned alpha: {node.alpha:.3f}, beta: {node.beta_param:.3f}")
                    nodes_processed += 1

            # Add children to visit list
            nodes_to_visit.extend(node.children)

        print(f"Completed learning for {nodes_processed} internal nodes.")
        return root

    def create_tree_from_clustering(self, data: np.ndarray) -> TreeNode:
        """
        Create a binary tree structure using agglomerative clustering.
        """
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform

        n_categories = data.shape[1]

        # Compute similarity matrix using correlation
        proportions = data / (data.sum(axis=1, keepdims=True) + 1e-8)
        similarity = np.corrcoef(proportions.T)
        similarity = np.nan_to_num(similarity, 0)

        # Convert to distance matrix
        distance = 1 - similarity
        np.fill_diagonal(distance, 0)
        distance = (distance + distance.T) / 2

        # Perform hierarchical clustering
        condensed_dist = squareform(distance)
        linkage_matrix = linkage(condensed_dist, method='ward')

        # Optional: Plot dendrogram
        try:
            from matplotlib import pyplot as plt
            dendrogram(linkage_matrix, labels=list(topic_labels.values())[1:], orientation='top')
            plt.tight_layout()
            plt.show()
        except ImportError:
            pass  # Skip plotting if matplotlib not available

        def build_tree_recursive(children_indices):
            """Recursively build binary tree from clustering."""
            if len(children_indices) == 1:
                return self._create_leaf_node(children_indices[0])

            if len(children_indices) == 2:
                child1 = self._create_leaf_node(children_indices[0])
                child2 = self._create_leaf_node(children_indices[1])
                return self._create_internal_node([child1, child2])

            # For more than 2 categories, split into exactly 2 groups using clustering
            subset_distance = distance[np.ix_(children_indices, children_indices)]
            subset_condensed = squareform(subset_distance)
            subset_linkage = linkage(subset_condensed, method='ward')

            # Split into exactly 2 clusters
            from scipy.cluster.hierarchy import fcluster
            cluster_labels = fcluster(subset_linkage, 2, criterion='maxclust')

            cluster1 = [children_indices[i] for i in range(len(children_indices)) if cluster_labels[i] == 1]
            cluster2 = [children_indices[i] for i in range(len(children_indices)) if cluster_labels[i] == 2]

            if len(cluster1) == 0 or len(cluster2) == 0:
                # Fallback split
                mid = len(children_indices) // 2
                cluster1 = children_indices[:mid]
                cluster2 = children_indices[mid:]

            child1 = build_tree_recursive(cluster1)
            child2 = build_tree_recursive(cluster2)

            return self._create_internal_node([child1, child2])

        return build_tree_recursive(list(range(n_categories)))

    def print_tree(self, node: TreeNode, depth: int = 0) -> None:
        """Print tree structure with learned parameters including p[i] statistics."""
        indent = "  " * depth
        if node.is_leaf:
            print(f"{indent}Leaf {node.node_id}: category {list(node.leaf_indices)}")
        else:
            alpha_str = ""
            if node.alpha is not None:
                alpha_str = f", α={node.alpha:.3f}, β={node.beta_param:.3f}"

            p_str = ""
            if node.p_values is not None:
                p_mean = np.mean(node.p_values)
                p_std = np.std(node.p_values)
                p_str = f", p[i]: μ={p_mean:.3f}±{p_std:.3f}"

            print(f"{indent}Internal {node.node_id}: categories {sorted(node.leaf_indices)}{alpha_str}{p_str}")
            for child in node.children:
                self.print_tree(child, depth + 1)

    def compute_tree_log_likelihood(self, root: TreeNode, data: np.ndarray) -> float:
        """Compute total log likelihood of the tree with learned parameters including p[i] values."""
        total_log_likelihood = 0.0

        nodes_to_visit = [root]
        while nodes_to_visit:
            node = nodes_to_visit.pop(0)

            if (not node.is_leaf and len(node.children) == 2 and
                    node.alpha is not None and node.p_values is not None):

                # Get counts for this node
                counts = self._compute_node_counts(node, data)

                # Filter to valid samples (same as during training)
                valid_samples = counts.sum(axis=1) > 0
                if np.any(valid_samples):
                    counts_valid = counts[valid_samples]

                    # Use the stored p[i] values for valid samples
                    log_likelihood = self._log_likelihood_node(
                        counts_valid, node.p_values, node.alpha, node.beta_param
                    )
                    total_log_likelihood += log_likelihood

            nodes_to_visit.extend(node.children)

        return total_log_likelihood

    def compute_distribution_likelihood(self, root: TreeNode, distribution: np.ndarray) -> float:
        """
        Compute the log likelihood of a given distribution under the learned tree model.

        Args:
            root: Root node of the tree with learned alpha parameters
            distribution: Probability distribution over categories (shape: n_categories)
                         Must be non-negative and sum to 1

        Returns:
            Log likelihood of the distribution under the tree model
        """
        # Validate input distribution
        if not isinstance(distribution, np.ndarray):
            distribution = np.array(distribution)

        if np.any(distribution < 0):
            raise ValueError("Distribution must have non-negative values")

        if np.abs(np.sum(distribution) - 1.0) > 1e-10:
            raise ValueError("Distribution must sum to 1")

        # Compute density by traversing tree and applying Beta densities
        def compute_node_density(node: TreeNode, node_distribution: np.ndarray) -> float:
            """Recursively compute density for a node."""
            if node.is_leaf or len(node.children) != 2 or node.alpha is None:
                return 0.0

            # Get the portion of the distribution for each child (binary tree)
            left_child = node.children[0]
            right_child = node.children[1]

            left_total = np.sum(node_distribution[list(left_child.leaf_indices)])
            right_total = np.sum(node_distribution[list(right_child.leaf_indices)])

            total_mass = left_total + right_total

            # Skip if no probability mass
            if total_mass == 0:
                return 0.0

            # Get proportion going to left child
            left_prob = left_total / total_mass

            # Compute Beta density
            if left_prob <= 0 or left_prob >= 1:
                return -np.inf

            log_density = self._log_beta_pdf(left_prob, node.alpha, node.beta_param)

            # Recursively compute for children
            for i, child in enumerate([left_child, right_child]):
                if not child.is_leaf:
                    child_total = left_total if i == 0 else right_total
                    if child_total > 0:
                        # Get conditional distribution for this child
                        child_dist = np.zeros_like(node_distribution)
                        for leaf_idx in child.leaf_indices:
                            child_dist[leaf_idx] = node_distribution[leaf_idx] / child_total

                        log_density += compute_node_density(child, child_dist)

            return log_density

        return compute_node_density(root, distribution)

    def sample_distribution_from_tree(self, root: TreeNode, random_state: Optional[int] = None) -> np.ndarray:
        """
        Sample a random probability distribution from the learned tree model.

        This method generates a sample by:
        1. Traversing the tree from root to leaves
        2. At each internal node, sampling from Beta(alpha, beta) to get split probability
        3. Recursively allocating probability mass to children
        4. Returning the final probability distribution over categories

        Args:
            root: Root node of the learned tree
            random_state: Optional random seed for reproducibility

        Returns:
            np.ndarray: Probability distribution over categories (sums to 1.0)
        """
        if random_state is not None:
            np.random.seed(random_state)

        # Get number of categories from leaf indices
        all_leaf_indices = set()

        def collect_leaves(node):
            if node.is_leaf:
                all_leaf_indices.update(node.leaf_indices)
            else:
                for child in node.children:
                    collect_leaves(child)

        collect_leaves(root)
        n_categories = len(all_leaf_indices)
        max_category = max(all_leaf_indices)

        # Initialize probability distribution
        distribution = np.zeros(max_category + 1)

        def sample_node_recursive(node: TreeNode, available_mass: float) -> None:
            """
            Recursively sample probability mass allocation for a node.

            Args:
                node: Current tree node
                available_mass: Total probability mass available to this node
            """
            if node.is_leaf:
                # Assign all available mass to this category
                category_idx = list(node.leaf_indices)[0]
                distribution[category_idx] += available_mass
                return

            if len(node.children) != 2:
                # Fallback: distribute equally among children
                mass_per_child = available_mass / len(node.children)
                for child in node.children:
                    sample_node_recursive(child, mass_per_child)
                return

            # Sample split probability from learned beta distribution
            if node.alpha is not None and node.beta_param is not None:
                # Sample from Beta(alpha, beta)
                left_prob = np.random.beta(node.alpha, node.beta_param)
            else:
                # Fallback: uniform split if no learned parameters
                left_prob = 0.5

            # Allocate mass to children
            left_mass = available_mass * left_prob
            right_mass = available_mass * (1 - left_prob)

            # Recursively sample for children
            sample_node_recursive(node.children[0], left_mass)
            sample_node_recursive(node.children[1], right_mass)

        # Start sampling from root with full probability mass
        sample_node_recursive(root, 1.0)

        # Trim to actual number of categories and ensure it sums to 1
        distribution = distribution[:n_categories]

        # Normalize to handle any numerical errors
        total_mass = np.sum(distribution)
        if total_mass > 0:
            distribution = distribution / total_mass
        else:
            # Fallback to uniform if something went wrong
            distribution = np.ones(n_categories) / n_categories

        return distribution


def generate_sample_data(n_samples: int = 1000, n_categories: int = 8) -> np.ndarray:
    """Generate sample count data with hierarchical structure."""
    np.random.seed(42)
    data = np.zeros((n_samples, n_categories))

    for i in range(n_samples):
        # Choose a group
        group = np.random.randint(0, n_categories // 2)
        total_count = np.random.poisson(50)
        group_prob = 0.7

        for _ in range(total_count):
            if np.random.random() < group_prob:
                cat = group * 2 + np.random.randint(0, 2)
            else:
                cat = np.random.randint(0, n_categories)
            data[i, cat] += 1

    return data.astype(int)


# Example usage
if __name__ == "__main__":
    # Load data or generate sample data
    target_topic = 266
    data = np.load(f"output/author_topic_{target_topic}.npy")
    print("Loaded data from file")
    print(f"Data shape: {data.shape}")

    # # Subsample if too large for demo
    # if data.shape[0] > 1000:
    #     indices = np.random.choice(data.shape[0], 1000, replace=False)
    #     data = data[indices]
    #     print(f"Subsampled to: {data.shape}")

    print(f"Data shape: {data.shape}")
    print(f"Sample counts per category: {np.mean(data, axis=0)}")

    # Create learner
    learner = SimplifiedDirichletTreeLearner()

    # Create tree structure from agglomerative clustering
    tree = learner.create_tree_from_clustering(data)
    print("\nInitial tree structure:")
    learner.print_tree(tree)

    # Learn alpha parameters
    tree_with_params = learner.learn_tree_parameters(tree, data)

    with open(f'output/portfolio_distribution_tree_{target_topic}.pickle', 'wb') as output:
        pickle.dump(tree_with_params, output, pickle.HIGHEST_PROTOCOL)

    print("\nTree with learned alpha parameters:")
    learner.print_tree(tree_with_params)

    # Compute final log likelihood
    final_log_likelihood = learner.compute_tree_log_likelihood(tree_with_params, data)
    print(f"\nFinal tree log likelihood: {final_log_likelihood:.2f}")

    # Sample a single distribution
    sampled_dist = learner.sample_distribution_from_tree(tree_with_params, random_state=42)
    print(f"Sampled distribution: {sampled_dist}")
    print(f"Sampled distribution sums to: {np.sum(sampled_dist):.6f}")

    indices = np.argpartition(sampled_dist, -16)[-16:]
    for idx in indices:
        print(f"{topic_labels[idx]}: {sampled_dist[idx]:.3f}")

    # Example: Compute likelihood of a test distribution
    print("\n" + "=" * 50)
    print("Testing distribution likelihood computation:")

    # Create a test distribution (normalize some sample from data)
    test_distribution = data[0].astype(float)  # Take first sample
    test_distribution = test_distribution / np.sum(test_distribution)  # Normalize to probabilities

    print(f"Test distribution: {test_distribution}")
    print(f"Distribution sums to: {np.sum(test_distribution):.6f}")

    # Compute likelihood using the learned model
    log_likelihood = learner.compute_distribution_likelihood(tree_with_params, test_distribution)
    print(f"Log likelihood of test distribution: {log_likelihood:.2f}")

    # Test with uniform distribution
    n_categories = data.shape[1]
    uniform_dist = np.ones(n_categories) / n_categories  # Properly normalized uniform distribution
    print(f"\nUniform distribution: {uniform_dist}")

    try:
        uniform_likelihood = learner.compute_distribution_likelihood(tree_with_params, uniform_dist)
        print(f"Log likelihood of uniform distribution: {uniform_likelihood:.2f}")
    except Exception as e:
        print(f"Error computing uniform likelihood: {e}")
