import pandas as pd
import numpy as np
from scipy.special import gammaln, loggamma
from typing import List, Tuple, Dict, Optional, Set
import itertools
from dataclasses import dataclass
from copy import deepcopy
import random
import numba
from numba import jit, njit, prange
from cmdstanpy import CmdStanModel

model = CmdStanModel(stan_file='code/fit.stan')

@dataclass
class TreeNode:
    """Represents a node in the Dirichlet tree."""
    node_id: int
    children: List['TreeNode']
    parent: Optional['TreeNode'] = None
    is_leaf: bool = False
    leaf_indices: Set[int] = None  # For leaf nodes, which original categories they represent

    def __post_init__(self):
        if self.leaf_indices is None:
            self.leaf_indices = set()


@njit
def gammaln_numba(x):
    """Numba-compatible log gamma function using Stirling's approximation for speed."""
    # For small values, use a lookup table or series expansion
    if x < 12.0:
        # Use Lanczos approximation for better accuracy
        g = 7
        coeff = np.array([
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7
        ])

        z = x - 1.0
        x_temp = coeff[0]
        for i in range(1, len(coeff)):
            x_temp += coeff[i] / (z + i)

        t = z + g + 0.5
        return (z + 0.5) * np.log(t) - t + 0.5 * np.log(2.0 * np.pi) + np.log(x_temp)
    else:
        # Stirling's approximation for large values
        return (x - 0.5) * np.log(x) - x + 0.5 * np.log(2.0 * np.pi)


# === KEY OPTIMIZATION 1: Cache evidence computations ===
class EvidenceCache:
    """Cache for storing computed evidence values to avoid recomputation."""

    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def get_key(self, leaf_indices_tuple, n_children):
        """Create cache key from node configuration."""
        return (leaf_indices_tuple, n_children)

    def get(self, leaf_indices, n_children):
        """Get cached evidence if available."""
        key = self.get_key(tuple(sorted(leaf_indices)), n_children)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, leaf_indices, n_children, evidence):
        """Cache evidence value."""
        key = self.get_key(tuple(sorted(leaf_indices)), n_children)
        self.cache[key] = evidence

    def stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return f"Cache: {self.hits}/{total} hits ({hit_rate:.1%})"


# === KEY OPTIMIZATION 2: Precompute node data once ===
@njit
def precompute_all_node_data_numba(data, leaf_indices_list):
    """
    Precompute count data for all possible node configurations.
    This avoids recomputing the same sums repeatedly.
    """
    n_samples = data.shape[0]
    n_configs = len(leaf_indices_list)

    # Precompute sums for each leaf set
    precomputed = np.zeros((n_configs, n_samples))

    for config_idx in range(n_configs):
        leaf_indices = leaf_indices_list[config_idx]
        for sample_idx in range(n_samples):
            total = 0
            for leaf_idx in leaf_indices:
                total += data[sample_idx, leaf_idx]
            precomputed[config_idx, sample_idx] = total

    return precomputed


# === KEY OPTIMIZATION 3: Faster tree traversal with iteration instead of recursion ===
def compute_tree_evidence_optimized(root, data, cache, precomputed_data=None):
    """
    Optimized tree evidence computation with caching and precomputed data.
    """
    total_evidence = 0.0

    # Use iterative traversal instead of recursive
    nodes_to_visit = [root]

    while nodes_to_visit:
        node = nodes_to_visit.pop()  # Use stack (LIFO) for efficiency

        if not node.is_leaf and len(node.children) > 1:
            # Check cache first
            cached_evidence = cache.get(node.leaf_indices, len(node.children))
            if cached_evidence is not None:
                total_evidence += cached_evidence
            else:
                # Compute evidence and cache it
                if precomputed_data is not None:
                    # Use precomputed data if available
                    node_counts = get_node_counts_from_precomputed(node, precomputed_data)
                else:
                    # Fallback to original computation
                    node_counts = compute_node_counts_optimized(node, data)

                alpha = np.full(len(node.children), 1.0)  # Use fixed alpha_prior

                # Prepare your data (replace with your actual data)
                n = node_counts[node_counts.sum(axis=1)>1][:200]
                data = {
                    'N': n.shape[0],
                    'M': n.shape[1],
                    'n': n.astype(int)
                }

                print(n)

                # Find MLE using optimization
                mle_fit = model.optimize(
                    data=data,
                    # algorithm='newton',
                    # iter=1000
                    # iter=2000,
                    # init_alpha=0.1,
                    # tol_obj=1e-12,
                    # tol_rel_obj=1e4,
                    # tol_grad=1e-8,
                    # tol_rel_grad=1e7,
                    # tol_param=1e-8
                )

                # Retrieve MLE estimates
                mle_estimates = mle_fit.stan_variables()
                print(mle_estimates['p'])

                evidence = compute_dirichlet_evidence_numba(node_counts, alpha)
                cache.set(node.leaf_indices, len(node.children), evidence)
                total_evidence += evidence

        # Add children to stack (internal nodes only)
        for child in node.children:
            if not child.is_leaf:
                nodes_to_visit.append(child)

    return total_evidence


@njit
def compute_node_counts_optimized_numba(data, child_leaf_indices_flat, child_sizes, n_children):
    """
    Ultra-optimized node count computation using flattened arrays.
    """
    n_samples = data.shape[0]
    child_counts = np.zeros((n_samples, n_children))

    start_idx = 0
    for child_idx in range(n_children):
        child_size = child_sizes[child_idx]

        for sample_idx in range(n_samples):
            total = 0
            for i in range(start_idx, start_idx + child_size):
                leaf_idx = child_leaf_indices_flat[i]
                total += data[sample_idx, leaf_idx]
            child_counts[sample_idx, child_idx] = total

        start_idx += child_size

    return child_counts


def compute_node_counts_optimized(node, data):
    """Optimized node count computation."""
    if node.is_leaf:
        return data[:, list(node.leaf_indices)[0]]

    # Flatten child indices for Numba
    child_leaf_indices_flat = []
    child_sizes = []

    for child in node.children:
        child_indices = list(child.leaf_indices)
        child_leaf_indices_flat.extend(child_indices)
        child_sizes.append(len(child_indices))

    child_leaf_indices_flat = np.array(child_leaf_indices_flat, dtype=np.int64)
    child_sizes = np.array(child_sizes, dtype=np.int64)

    return compute_node_counts_optimized_numba(
        data, child_leaf_indices_flat, child_sizes, len(node.children)
    )


# === KEY OPTIMIZATION 4: Batch validation ===
def validate_tree_fast(root, n_categories):
    """Fast tree validation using sets."""
    all_leaves = set()
    stack = [root]

    while stack:
        node = stack.pop()
        if node.is_leaf:
            all_leaves.update(node.leaf_indices)
        else:
            stack.extend(node.children)

    return all_leaves == set(range(n_categories))


# === KEY OPTIMIZATION 5: Reduce deep copying ===
def deep_copy_tree_optimized(root):
    """Optimized tree copying using iteration."""
    old_to_new = {}
    stack = [(root, None)]  # (node, new_parent)
    new_root = None

    while stack:
        old_node, new_parent = stack.pop()

        if old_node.node_id in old_to_new:
            new_node = old_to_new[old_node.node_id]
        else:
            # Create new node
            new_node = TreeNode(
                node_id=old_node.node_id,
                children=[],
                is_leaf=old_node.is_leaf,
                leaf_indices=old_node.leaf_indices.copy()
            )
            old_to_new[old_node.node_id] = new_node

            if new_root is None:
                new_root = new_node

        # Set parent
        if new_parent is not None:
            new_node.parent = new_parent
            new_parent.children.append(new_node)

        # Add children to stack
        for child in old_node.children:
            stack.append((child, new_node))

    return new_root

@njit
def compute_dirichlet_evidence_numba(counts, alpha):
    """
    Numba-optimized computation of Dirichlet-multinomial evidence.

    Args:
        counts: Count matrix (n_samples x n_categories)
        alpha: Dirichlet prior parameters (1D array)

    Returns:
        Log evidence (scalar)
    """
    n_samples, n_categories = counts.shape
    log_evidence = 0.0
    alpha_sum = np.sum(alpha)

    for i in range(n_samples):
        total_count = 0
        for j in range(n_categories):
            total_count += counts[i, j]

        # Log multinomial coefficient
        log_evidence += gammaln_numba(total_count + 1)
        for j in range(n_categories):
            log_evidence -= gammaln_numba(counts[i, j] + 1)

        # Dirichlet-multinomial terms
        log_evidence += gammaln_numba(alpha_sum) - gammaln_numba(alpha_sum + total_count)
        for j in range(n_categories):
            log_evidence += gammaln_numba(alpha[j] + counts[i, j]) - gammaln_numba(alpha[j])

    return log_evidence


@njit
def compute_node_counts_numba(data, leaf_indices_list, child_leaf_indices):
    """
    Numba-optimized computation of node counts.

    Args:
        data: Full count matrix (n_samples x n_categories)
        leaf_indices_list: List of leaf indices for the node (not used but kept for compatibility)
        child_leaf_indices: List of arrays, each containing leaf indices for a child

    Returns:
        Count matrix for this node (n_samples x n_children)
    """
    n_samples = data.shape[0]
    n_children = len(child_leaf_indices)

    child_counts = np.zeros((n_samples, n_children))

    for child_idx in range(n_children):
        leaf_indices = child_leaf_indices[child_idx]
        for sample_idx in range(n_samples):
            total = 0
            for leaf_idx in leaf_indices:
                total += data[sample_idx, leaf_idx]
            child_counts[sample_idx, child_idx] = total

    return child_counts


@njit
def sum_counts_for_indices_numba(data, indices):
    """Sum counts for specific category indices across all samples."""
    n_samples = data.shape[0]
    result = np.zeros(n_samples)

    for i in range(n_samples):
        total = 0
        for idx in indices:
            total += data[i, idx]
        result[i] = total

    return result


class DirichletTreeLearner:
    """
    Learns Dirichlet-tree structure from count data using MCMC with local moves.

    Based on Minka (1999) "The Dirichlet-tree distribution"
    """

    def __init__(self, alpha_prior: float = 1.0):
        """
        Initialize the learner.

        Args:
            alpha_prior: Prior parameter for Dirichlet distributions at each node
        """
        self.alpha_prior = alpha_prior
        self.next_node_id = 0

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
        """Create an internal node with given children."""
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
        Compute count data for a specific node using Numba optimization.
        """
        if node.is_leaf:
            # For leaf nodes, return the counts for that category
            return data[:, list(node.leaf_indices)[0]]

        # For internal nodes, prepare data for Numba function
        child_leaf_indices = []
        for child in node.children:
            child_indices = np.array(list(child.leaf_indices), dtype=np.int64)
            child_leaf_indices.append(child_indices)

        # Use Numba-optimized function
        # Note: We need to handle the list of arrays carefully for Numba
        child_counts = []
        for child_indices in child_leaf_indices:
            child_total = sum_counts_for_indices_numba(data, child_indices)
            child_counts.append(child_total)

        return np.array(child_counts).T  # Shape: (n_samples, n_children)

    def _compute_dirichlet_evidence(self, counts: np.ndarray, alpha: np.ndarray) -> float:
        """
        Compute the log evidence for a Dirichlet-multinomial model using Numba.
        """
        # Ensure inputs are the right type for Numba
        counts = counts.astype(np.float64)
        alpha = alpha.astype(np.float64)

        return compute_dirichlet_evidence_numba(counts, alpha)

    #
    # def _compute_node_counts(self, node: TreeNode, data: np.ndarray) -> np.ndarray:
    #     """
    #     Compute count data for a specific node.
    #
    #     For an internal node, this gives the counts for each child branch.
    #
    #     Args:
    #         node: The tree node
    #         data: Count matrix (n_samples x n_categories)
    #
    #     Returns:
    #         Array of counts for each child of this node
    #     """
    #     if node.is_leaf:
    #         # For leaf nodes, return the counts for that category
    #         return data[:, list(node.leaf_indices)[0]]
    #
    #     # For internal nodes, sum counts for each child's subtree
    #     child_counts = []
    #     for child in node.children:
    #         child_total = np.sum(data[:, list(child.leaf_indices)], axis=1)
    #         child_counts.append(child_total)
    #
    #     return np.array(child_counts).T  # Shape: (n_samples, n_children)
    #
    # def _compute_dirichlet_evidence(self, counts: np.ndarray, alpha: np.ndarray) -> float:
    #     """
    #     Compute the log evidence for a Dirichlet-multinomial model.
    #
    #     Args:
    #         counts: Count matrix (n_samples x n_categories)
    #         alpha: Dirichlet prior parameters
    #
    #     Returns:
    #         Log evidence
    #     """
    #     n_samples, n_categories = counts.shape
    #
    #     # Total counts per sample
    #     total_counts = np.sum(counts, axis=1)
    #
    #     # Log evidence calculation
    #     log_evidence = 0.0
    #
    #     for i in range(n_samples):
    #         # Log multinomial coefficient
    #         log_evidence += gammaln(total_counts[i] + 1)
    #         log_evidence -= np.sum(gammaln(counts[i] + 1))
    #
    #         # Log Dirichlet-multinomial terms
    #         log_evidence += gammaln(np.sum(alpha)) - gammaln(np.sum(alpha) + total_counts[i])
    #         log_evidence += np.sum(gammaln(alpha + counts[i]) - gammaln(alpha))
    #
    #     return log_evidence

    def _compute_tree_evidence(self, root: TreeNode, data: np.ndarray) -> float:
        """
        Compute the total log evidence for a tree structure.

        Args:
            root: Root node of the tree
            data: Count matrix (n_samples x n_categories)

        Returns:
            Total log evidence
        """
        total_evidence = 0.0

        # Traverse all internal nodes
        nodes_to_visit = [root]
        while nodes_to_visit:
            node = nodes_to_visit.pop(0)

            if not node.is_leaf and len(node.children) > 1:
                # Compute evidence for this node
                node_counts = self._compute_node_counts(node, data)
                alpha = np.full(len(node.children), self.alpha_prior)
                evidence = self._compute_dirichlet_evidence(node_counts, alpha)
                total_evidence += evidence

            # Add children to visit list
            nodes_to_visit.extend([child for child in node.children if not child.is_leaf])

        return total_evidence

    def _get_all_internal_nodes(self, root: TreeNode) -> List[TreeNode]:
        """Get all internal nodes in the tree."""
        internal_nodes = []
        nodes_to_visit = [root]

        while nodes_to_visit:
            node = nodes_to_visit.pop(0)
            if not node.is_leaf:
                internal_nodes.append(node)
                nodes_to_visit.extend([child for child in node.children if not child.is_leaf])

        return internal_nodes

    def _deep_copy_tree(self, root: TreeNode) -> TreeNode:
        """Create a deep copy of the tree structure."""
        # Create mapping from old nodes to new nodes
        old_to_new = {}

        def copy_node(old_node):
            if old_node.node_id in old_to_new:
                return old_to_new[old_node.node_id]

            # Create new node
            new_node = TreeNode(
                node_id=old_node.node_id,
                children=[],
                is_leaf=old_node.is_leaf,
                leaf_indices=old_node.leaf_indices.copy()
            )
            old_to_new[old_node.node_id] = new_node

            # Copy children
            for child in old_node.children:
                new_child = copy_node(child)
                new_node.children.append(new_child)
                new_child.parent = new_node

            return new_node

        return copy_node(root)

    def _propose_subtree_prune_regraft(self, root: TreeNode) -> Optional[TreeNode]:
        """
        Propose a subtree prune-and-regraft (SPR) move.

        1. Choose a random subtree to prune (not root)
        2. Choose a random location to regraft it
        3. Return the modified tree
        """
        # Copy the tree
        new_root = self._deep_copy_tree(root)

        # Get all nodes that can be pruned (any node except root)
        all_nodes = []
        nodes_to_visit = [new_root]
        while nodes_to_visit:
            node = nodes_to_visit.pop(0)
            if node.parent is not None:  # Not root
                all_nodes.append(node)
            nodes_to_visit.extend(node.children)

        if not all_nodes:
            return None

        # Choose a node to prune
        prune_node = random.choice(all_nodes)
        prune_parent = prune_node.parent

        # Don't prune if it would leave parent with only 1 child (unless parent is root)
        if len(prune_parent.children) <= 2 and prune_parent.parent is not None:
            return None

        # Remove the pruned node from its parent
        prune_parent.children.remove(prune_node)
        prune_node.parent = None

        # Get all possible regraft locations (internal nodes, excluding pruned subtree)
        regraft_candidates = []
        nodes_to_visit = [new_root]
        while nodes_to_visit:
            node = nodes_to_visit.pop(0)
            if not node.is_leaf and not self._is_descendant(node, prune_node):
                regraft_candidates.append(node)
            # Only visit children that are not in the pruned subtree
            for child in node.children:
                if not self._is_descendant(child, prune_node):
                    nodes_to_visit.append(child)

        if not regraft_candidates:
            # Restore the tree if no valid regraft location
            prune_parent.children.append(prune_node)
            prune_node.parent = prune_parent
            return None

        # Choose a location to regraft
        regraft_parent = random.choice(regraft_candidates)

        # Add the pruned subtree to the new location
        regraft_parent.children.append(prune_node)
        prune_node.parent = regraft_parent

        # Update leaf indices up the tree
        self._update_leaf_indices(new_root)

        return new_root

    def _is_descendant(self, potential_descendant: TreeNode, ancestor: TreeNode) -> bool:
        """Check if potential_descendant is in the subtree rooted at ancestor."""
        nodes_to_check = [ancestor]
        while nodes_to_check:
            node = nodes_to_check.pop(0)
            if node == potential_descendant:
                return True
            nodes_to_check.extend(node.children)
        return False

    def _propose_subtree_swap(self, root: TreeNode) -> Optional[TreeNode]:
        """
        Propose swapping two subtrees in the tree.
        """
        # Copy the tree
        new_root = self._deep_copy_tree(root)

        # Get all internal nodes with at least 2 children
        internal_nodes = self._get_all_internal_nodes(new_root)
        swap_candidates = [n for n in internal_nodes if len(n.children) >= 2]

        if len(swap_candidates) < 1:
            return None

        # Try to find two different parents, but allow same parent
        if len(swap_candidates) >= 2:
            node1, node2 = random.sample(swap_candidates, 2)
        else:
            # Only one candidate with >=2 children, try swapping within it
            node1 = swap_candidates[0]
            if len(node1.children) < 2:
                return None
            node2 = node1  # Swap within same parent

        # Choose a child from each node
        child1 = random.choice(node1.children)
        child2 = random.choice(node2.children)

        # Don't swap a node with itself
        if child1 == child2:
            return None

        # Don't swap if one is ancestor of the other
        if self._is_descendant(child1, child2) or self._is_descendant(child2, child1):
            return None

        # Perform the swap
        idx1 = node1.children.index(child1)
        idx2 = node2.children.index(child2)

        node1.children[idx1] = child2
        node2.children[idx2] = child1

        child1.parent = node2
        child2.parent = node1

        # Update leaf indices
        self._update_leaf_indices(new_root)

        return new_root

    def _propose_binary_split(self, root: TreeNode) -> Optional[TreeNode]:
        """
        Propose splitting an internal node with >2 children into two nodes.
        """
        # Copy the tree
        new_root = self._deep_copy_tree(root)

        # Find nodes with >2 children
        internal_nodes = self._get_all_internal_nodes(new_root)
        split_candidates = [n for n in internal_nodes if len(n.children) > 2]

        if not split_candidates:
            return None

        # Choose a node to split
        split_node = random.choice(split_candidates)
        original_children = split_node.children.copy()  # Make a copy

        # Randomly partition children into two non-empty groups
        n_children = len(original_children)
        if n_children < 3:
            return None

        # Create random binary partition ensuring both groups are non-empty
        indices = list(range(n_children))
        random.shuffle(indices)
        split_point = random.randint(1, n_children - 1)  # 1 to n-1

        partition1_indices = indices[:split_point]
        partition2_indices = indices[split_point:]

        partition1 = [original_children[i] for i in partition1_indices]
        partition2 = [original_children[i] for i in partition2_indices]

        # Create new internal nodes for multi-child partitions
        if len(partition1) == 1:
            new_child1 = partition1[0]
        else:
            new_child1 = TreeNode(
                node_id=self._get_next_node_id(),
                children=partition1.copy(),
                is_leaf=False,
                leaf_indices=set()
            )
            # Set parent relationships and collect leaf indices
            for child in partition1:
                child.parent = new_child1
                new_child1.leaf_indices.update(child.leaf_indices)

        if len(partition2) == 1:
            new_child2 = partition2[0]
        else:
            new_child2 = TreeNode(
                node_id=self._get_next_node_id(),
                children=partition2.copy(),
                is_leaf=False,
                leaf_indices=set()
            )
            # Set parent relationships and collect leaf indices
            for child in partition2:
                child.parent = new_child2
                new_child2.leaf_indices.update(child.leaf_indices)

        # Replace split_node's children with the two new groups
        split_node.children = [new_child1, new_child2]
        new_child1.parent = split_node
        new_child2.parent = split_node

        # Update leaf indices up the tree
        self._update_leaf_indices(new_root)

        return new_root

    def _validate_tree(self, root: TreeNode, n_categories: int) -> bool:
        """
        Validate that tree contains all categories exactly once.

        Args:
            root: Root of tree to validate
            n_categories: Expected number of categories

        Returns:
            True if tree is valid, False otherwise
        """
        # Collect all leaf indices
        all_leaves = set()
        leaf_nodes = []
        nodes_to_visit = [root]

        while nodes_to_visit:
            node = nodes_to_visit.pop(0)
            if node.is_leaf:
                all_leaves.update(node.leaf_indices)
                leaf_nodes.append(node)
            else:
                nodes_to_visit.extend(node.children)

        # Check if we have exactly the expected categories
        expected_categories = set(range(n_categories))

        if all_leaves != expected_categories:
            missing = expected_categories - all_leaves
            extra = all_leaves - expected_categories
            print(f"TREE VALIDATION FAILED!")
            print(f"  Expected {n_categories} categories: {sorted(expected_categories)}")
            print(f"  Found {len(all_leaves)} categories: {sorted(all_leaves)}")
            print(f"  Missing categories: {sorted(missing)}")
            print(f"  Extra categories: {sorted(extra)}")
            print(f"  Number of leaf nodes: {len(leaf_nodes)}")

            # Print first few leaf nodes for debugging
            print("  First few leaf nodes:")
            for i, leaf in enumerate(leaf_nodes[:10]):
                print(f"    Leaf {leaf.node_id}: {sorted(leaf.leaf_indices)}")
            if len(leaf_nodes) > 10:
                print(f"    ... and {len(leaf_nodes) - 10} more")

            return False

        return True
        """Update leaf indices for all nodes in the tree."""
        if node.is_leaf:
            return

        # Update children first
        for child in node.children:
            self._update_leaf_indices(child)

        # Update this node's leaf indices
        leaf_indices = set()
        for child in node.children:
            leaf_indices.update(child.leaf_indices)
        node.leaf_indices = leaf_indices

    def _update_leaf_indices(self, node: TreeNode):
        """Update leaf indices for all nodes in the tree."""
        if node.is_leaf:
            return

        # Update children first
        for child in node.children:
            self._update_leaf_indices(child)

        # Update this node's leaf indices
        leaf_indices = set()
        for child in node.children:
            leaf_indices.update(child.leaf_indices)
        node.leaf_indices = leaf_indices

    def _initialize_random_tree(self, n_categories: int) -> TreeNode:
        """Initialize a random binary tree structure."""
        # Create all leaf nodes
        leaves = [self._create_leaf_node(i) for i in range(n_categories)]

        # Randomly build binary tree
        active_nodes = leaves.copy()

        while len(active_nodes) > 1:
            # Randomly choose two nodes to merge
            if len(active_nodes) == 2:
                # Last merge
                child1, child2 = active_nodes
            else:
                child1, child2 = random.sample(active_nodes, 2)

            # Remove chosen nodes
            active_nodes.remove(child1)
            active_nodes.remove(child2)

            # Create new internal node
            new_node = self._create_internal_node([child1, child2])
            active_nodes.append(new_node)

        return active_nodes[0]

    def _compute_category_similarity(self, data: np.ndarray) -> np.ndarray:
        """Compute pairwise similarity between categories using correlation."""
        # Normalize data to get proportions
        proportions = data / (data.sum(axis=1, keepdims=True) + 1e-3)

        # Compute correlation matrix
        similarity = np.corrcoef(proportions.T)
        similarity = np.nan_to_num(similarity, 0)  # Handle NaN

        return (similarity + similarity.T) / 2

    def learn_structure_hierarchical(self, data: np.ndarray) -> TreeNode:
        """Learn structure using hierarchical clustering for initialization."""
        from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
        from scipy.spatial.distance import squareform
        from matplotlib import pyplot as plt

        n_categories = data.shape[1]

        print(n_categories)

        # Compute similarity and convert to distance
        similarity = self._compute_category_similarity(data)
        distance = 1 - similarity
        np.fill_diagonal(distance, 0)

        print(similarity.shape)

        # Hierarchical clustering
        condensed_dist = squareform(distance)
        linkage_matrix = linkage(condensed_dist, method='ward')

        print(linkage_matrix.shape)

        labels = list(topic_labels.values())[1:]
        dendrogram(linkage_matrix, labels=labels)

        # labels = list(topic_labels.values())[1:]
        # dendrogram(linkage_matrix, labels=list(range(n_categories)))
        plt.show()

        def build_tree_from_clustering(children_indices):
            if len(children_indices) == 1:
                return self._create_leaf_node(children_indices[0])

            if len(children_indices) == 2:
                # Base case: create internal node with two leaf children
                child1 = self._create_leaf_node(children_indices[0])
                child2 = self._create_leaf_node(children_indices[1])
                return self._create_internal_node([child1, child2])

            # Extract distance matrix for current subset of categories
            subset_distance = distance[np.ix_(children_indices, children_indices)]

            # Perform hierarchical clustering on this subset
            condensed_dist = squareform(subset_distance)
            subset_linkage = linkage(condensed_dist, method='ward')

            # Split into 2 clusters
            cluster_labels = fcluster(subset_linkage, 2, criterion='maxclust')

            # Group children by cluster
            cluster1 = [children_indices[i] for i in range(len(children_indices)) if cluster_labels[i] == 1]
            cluster2 = [children_indices[i] for i in range(len(children_indices)) if cluster_labels[i] == 2]

            if len(cluster1) == 0 or len(cluster2) == 0:
                # Fallback: split in half
                mid = len(children_indices) // 2
                cluster1 = children_indices[:mid]
                cluster2 = children_indices[mid:]

            # Recursively build subtrees
            child1 = build_tree_from_clustering(cluster1)
            child2 = build_tree_from_clustering(cluster2)

            return self._create_internal_node([child1, child2])

        # Optional: Show dendrogram for full dataset
        condensed_dist = squareform(distance)
        linkage_matrix = linkage(condensed_dist, method='ward')

        # labels = [topic_labels.get(i, f'Topic_{i}') for i in range(n_categories)]
        # dendrogram(linkage_matrix, labels=labels)
        # plt.show()

        # Build tree starting with all categories
        return build_tree_from_clustering(list(range(n_categories)))

    # === MAIN OPTIMIZED MCMC FUNCTION ===
    def learn_structure_mcmc(self, data, n_iterations=10000, burn_in=1000):
        """
        Highly optimized MCMC with multiple speedup techniques.
        """
        n_samples, n_categories = data.shape

        # Initialize cache
        cache = EvidenceCache()

        # Initialize tree
        current_tree = self.learn_structure_hierarchical(data)
        # current_tree = self._initialize_random_tree(n_categories)

        self.print_tree(current_tree)

        current_evidence = compute_tree_evidence_optimized(current_tree, data, cache)

        # Validate initial tree
        if not validate_tree_fast(current_tree, n_categories):
            raise ValueError("Initial tree is invalid!")

        # Track best tree
        best_tree = deep_copy_tree_optimized(current_tree)
        best_evidence = current_evidence

        print(f"Initial tree evidence: {current_evidence:.2f}")

        # MCMC sampling with optimizations
        samples = []
        n_accepted = 0
        n_proposed = 0
        n_invalid = 0

        # Pre-allocate random numbers to reduce overhead
        move_types = np.random.choice(['swap', 'split', 'spr'], size=n_iterations, p=[0.5, 0.25, 0.25])
        accept_randoms = np.random.random(n_iterations)

        for iteration in range(n_iterations):
            move_type = move_types[iteration]

            # Propose move
            if move_type == 'spr':
                proposed_tree = self._propose_subtree_prune_regraft(current_tree)
            elif move_type == 'swap':
                proposed_tree = self._propose_subtree_swap(current_tree)
            else:  # split
                proposed_tree = self._propose_binary_split(current_tree)

            if proposed_tree is None:
                continue

            # Fast validation
            if not validate_tree_fast(proposed_tree, n_categories):
                n_invalid += 1
                continue

            n_proposed += 1

            # Compute evidence with caching
            proposed_evidence = compute_tree_evidence_optimized(proposed_tree, data, cache)

            # Accept or reject
            log_ratio = proposed_evidence - current_evidence

            if log_ratio > 0 or np.log(accept_randoms[iteration]) < log_ratio:
                # Accept
                current_tree = proposed_tree
                current_evidence = proposed_evidence
                n_accepted += 1

                # Update best tree
                if current_evidence > best_evidence:
                    best_evidence = current_evidence
                    best_tree = deep_copy_tree_optimized(current_tree)

            # Collect sample after burn-in
            if iteration >= burn_in:
                samples.append(deep_copy_tree_optimized(current_tree))

            # Progress reporting (less frequent to reduce overhead)
            if iteration % 500 == 0 and iteration > 0:
                acceptance_rate = n_accepted / n_proposed if n_proposed > 0 else 0
                print(f"Iteration {iteration}: Current = {current_evidence:.2f}, "
                      f"Best = {best_evidence:.2f}, Accept rate = {acceptance_rate:.3f}")
                print(f"  {cache.stats()}")

        final_acceptance_rate = n_accepted / n_proposed if n_proposed > 0 else 0
        print(f"Final acceptance rate: {final_acceptance_rate:.3f}")
        print(f"Final {cache.stats()}")

        print(f"Total proposed moves: {n_proposed}")
        print(f"Invalid moves rejected: {n_invalid}")
        print(f"Collected {len(samples)} samples")
        print(f"Best evidence encountered: {best_evidence:.2f}")

        # Final validation
        if not self._validate_tree(best_tree, n_categories):
            raise ValueError("Best tree is invalid!")

        return samples, best_tree

    def print_tree(self, node: TreeNode, depth: int = 0) -> None:
        """Print tree structure."""
        indent = "  " * depth
        if node.is_leaf:
            print(
                f"{indent}Leaf {node.node_id}: categories {[topic_labels[topic] for topic in sorted(node.leaf_indices)]}")
        else:
            print(f"{indent}Internal {node.node_id}: categories {sorted(node.leaf_indices)}")
            for child in node.children:
                self.print_tree(child, depth + 1)


def generate_sample_data(n_samples: int = 1000, n_categories: int = 8) -> np.ndarray:
    """
    Generate sample count data with some hierarchical structure.

    Args:
        n_samples: Number of samples
        n_categories: Number of categories

    Returns:
        Count matrix (n_samples x n_categories)
    """
    np.random.seed(42)

    # Create data with some hierarchical clustering
    # Categories 0,1 are similar, 2,3 are similar, etc.
    data = np.zeros((n_samples, n_categories))

    for i in range(n_samples):
        # Choose a group (each pair of categories)
        super_group = np.random.randint(0, n_categories // 4)
        group = np.random.randint(0, n_categories // 2)

        # Sample total count for this sample
        total_count = np.random.poisson(4)

        # Distribute counts within the chosen group with some spillover
        group_prob = 0.7  # Probability of staying within group

        for _ in range(total_count):
            if np.random.random() < group_prob:
                # Stay within group
                cat = group * 2 + np.random.randint(0, 2)
            else:
                # Choose any category
                cat = np.random.randint(super_group*4, (super_group+1)*4)

            data[i, cat] += 1

    return data.astype(int)


# Example usage
if __name__ == "__main__":
    topic_labels = pd.read_csv("data/topic_list.csv").set_index("Topic")["Name"].to_dict()

    # Generate sample data
    data = np.load("output/author_topic_1.npy")
    data = data[np.random.choice(data.shape[0], 50000, replace=False)]

    print(topic_labels[data.sum(axis=0).argmax()])


    print(f"Data shape: {data.shape}")
    print(f"Sample counts per category: {np.mean(data, axis=0)}")
    print()

    # Learn tree structure using MCMC
    learner = DirichletTreeLearner(alpha_prior=1.0)
    samples, best_tree_from_mcmc = learner.learn_structure_mcmc(data, n_iterations=10000, burn_in=200)

    # Use the best tree from MCMC run
    print("\nBest tree found during MCMC:")
    learner.print_tree(best_tree_from_mcmc)

    # Print some statistics
    print(f"\nData statistics:")
    print(f"Total samples: {data.shape[0]}")
    print(f"Total categories: {data.shape[1]}")
    print(f"Mean counts per sample: {np.mean(np.sum(data, axis=1)):.1f}")
    print(f"Category frequencies: {np.sum(data, axis=0)}")
