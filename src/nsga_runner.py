"""
This module defines the NSGARunner class,
which is used to run the NSGA-II algorithm
to find the optimal set of refactorings
for the given source code.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
import random

@dataclass
class RefactoringPlan:
    """
    Represents a candidate solution = a sequence of refactoring genes.
    For now it's just a list[Any]
    Can be modified later (e.g. (op_type, ast_address, params)).
    """
    genes: List[Any]


@dataclass
class Individual:
    """
    Wraps a RefactoringPlan with its objective values and metadata.
    objectives: tuple of floats to minimize, e.g.
      (structural_score, cost_score, -readability_score)
    """
    plan: RefactoringPlan
    objectives: Optional[Tuple[float, ...]] = None
    rank: Optional[int] = None
    crowding_distance: float = 0.0

class NSGARunner:
    """
    Coordinates NSGA-II:
      - asks CandidateGenerator for initial plans
      - uses Applier to apply a plan to a codebase snapshot
      - calls MetricCalculator (+ ReadabilityScorer) to compute objectives
      - evolves a population and returns an approximate Pareto front
    """
    def __init__(
            self,
        candidate_generator: Callable[[], RefactoringPlan],
        applier: Callable[[RefactoringPlan], Any],
        metric_calculator: Callable[[Any], Dict[str, float]],
        readability_scorer: Optional[Callable[[Any], float]] = None,
        pop_size: int = 40,
        n_generations: int = 30,
        cx_prob: float = 0.7,
        mut_prob: float = 0.3,
        random_seed: int = 0,
        ):
            self.candidate_generator = candidate_generator
            self.applier = applier
            self.metric_calculator = metric_calculator
            self.readability_scorer = readability_scorer

            self.pop_size = pop_size
            self.n_generations = n_generations
            self.cx_prob = cx_prob
            self.mut_prob = mut_prob
            self.random = random.Random(random_seed)
        
    def run(self) -> List[Individual]:
        """
        Main entry point: run NSGA-II and return the final Pareto front.
        """
        population = self._init_population()
        self._evaluate_population(population)

        for gen in range(self.n_generations):
            print(f"[NSGA-II] Generation {gen+1}/{self.n_generations}")

            offspring = self._make_offspring(population)
            self._evaluate_population(offspring)

            population = self._environmental_selection(population + offspring)

        pareto_front = self._extract_pareto_front(population)
        return pareto_front
       
    def _init_population(self) -> List[Individual]:
            """
            Ask the candidate generator to create initial plans.
            """
            pop: List[Individual] = []
            for _ in range(self.pop_size):
                plan = self.candidate_generator()
                pop.append(Individual(plan=plan))
            return pop
    def _evaluate_population(self, population: List[Individual]) -> None:
        """
        For each individual:
          - apply its refactoring plan to a codebase snapshot
          - compute metrics
          - compute objectives tuple (to minimize)
        """
        for ind in population:
            if ind.objectives is not None:
                continue  # already evaluated (can be useful later)

            # Apply candidate refactorings
            transformed_code = self.applier(ind.plan)

            # Compute metrics (structural + cost)
            metrics = self.metric_calculator(transformed_code)

            # Example: combine into 2 structural/cost objectives
            structural_score = self._structural_objective(metrics)
            cost_score = self._cost_objective(metrics, ind.plan)

            # Readability (optional, we maximize it → use negative for minimization)
            if self.readability_scorer is not None:
                readability = self.readability_scorer(transformed_code)
                readability_obj = -float(readability)  # because NSGA-II minimizes
                objectives = (structural_score, cost_score, readability_obj)
            else:
                objectives = (structural_score, cost_score)

            ind.objectives = objectives

    def _structural_objective(self, metrics: Dict[str, float]) -> float:
        """
        Combine structural metrics into a single value to minimize.
        Weights to be adjusted after deciding which metrics to use.
        Example metrics keys for now: "avg_cc", "max_cc", "max_nesting", "avg_loc"
        """
        cc = metrics.get("avg_cc", 0.0)
        nesting = metrics.get("max_nesting", 0.0)
        loc = metrics.get("avg_loc", 0.0)
        return cc + nesting + loc
    
    def _cost_objective(self, metrics: Dict[str, float], plan: RefactoringPlan) -> float:
        """
        Cost / regularization objective: fewer changes, fewer failed ops, etc.
        Example metrics keys for now: "num_refactorings", "num_failed_ops"
        """
        num_refactorings = metrics.get("num_refactorings", len(plan.genes))
        num_failed = metrics.get("num_failed_ops", 0.0)
        return float(num_refactorings) + 10.0 * float(num_failed)

    def _make_offspring(self, population: List[Individual]) -> List[Individual]:
        """
        Create offspring using tournament selection, crossover, and mutation.
        For now the genetic operators are very simple. Will be refined
        once the representation is fixed.
        """
        offspring: List[Individual] = []

        while len(offspring) < self.pop_size:
            parent1 = self._tournament_select(population)
            parent2 = self._tournament_select(population)

            child1_plan = self._clone_plan(parent1.plan)
            child2_plan = self._clone_plan(parent2.plan)

            # Crossover
            if self.random.random() < self.cx_prob:
                child1_plan, child2_plan = self._crossover(child1_plan, child2_plan)

            # Mutation
            if self.random.random() < self.mut_prob:
                child1_plan = self._mutate(child1_plan)
            if self.random.random() < self.mut_prob:
                child2_plan = self._mutate(child2_plan)

            offspring.append(Individual(plan=child1_plan))
            if len(offspring) < self.pop_size:
                offspring.append(Individual(plan=child2_plan))

        return offspring
    
    def _clone_plan(self, plan: RefactoringPlan) -> RefactoringPlan:
        return RefactoringPlan(genes=list(plan.genes))
    
    def _tournament_select(self, population: List[Individual], k: int = 2) -> Individual:
        """
        Binary tournament based on Pareto rank then crowding distance.
        """
        candidates = self.random.sample(population, k)
        # Population is assumed to already have rank and crowding_distance set.
        # For now, if rank is None, we fall back to first objective.
        def key(ind: Individual):
            if ind.rank is None:
                return (0, ind.objectives[0] if ind.objectives is not None else float("inf"))
            return (ind.rank, -ind.crowding_distance)

        return min(candidates, key=key)

    def _crossover(
        self,
        plan1: RefactoringPlan,
        plan2: RefactoringPlan,
    ) -> Tuple[RefactoringPlan, RefactoringPlan]:
        """
        Simple one-point crossover on gene lists.
        Later will be improved (e.g., respect AST scopes).
        """
        if not plan1.genes or not plan2.genes:
            return plan1, plan2

        cut1 = self.random.randint(1, len(plan1.genes))
        cut2 = self.random.randint(1, len(plan2.genes))

        child1_genes = plan1.genes[:cut1] + plan2.genes[cut2:]
        child2_genes = plan2.genes[:cut2] + plan1.genes[cut1:]

        return RefactoringPlan(child1_genes), RefactoringPlan(child2_genes)

    def _mutate(self, plan: RefactoringPlan) -> RefactoringPlan:
        """
        Placeholder mutation: either drop a gene or (later) request a new gene
        from the candidate generator.
        To be refined once the representation is finalized.
        """
        genes = list(plan.genes)
        if not genes:
            # maybe add a new random gene in the future
            return plan

        # With 50% chance: remove a random gene
        if self.random.random() < 0.5:
            idx = self.random.randrange(len(genes))
            del genes[idx]
        # Otherwise: TODO: replace gene / slightly perturb parameters

        return RefactoringPlan(genes=genes)
    
    def _environmental_selection(self, population: List[Individual]) -> List[Individual]:
        """
        NSGA-II environmental selection:
          - non-dominated sorting
          - crowding distance
          - keep best pop_size individuals
        Can be refined later.
        """
        fronts = self._non_dominated_sort(population)
        new_pop: List[Individual] = []
        for rank, front in enumerate(fronts):
            self._assign_crowding_distance(front)
            for ind in front:
                ind.rank = rank
            if len(new_pop) + len(front) <= self.pop_size:
                new_pop.extend(front)
            else:
                # Sort this front by crowding distance descending and fill remaining slots
                remaining = self.pop_size - len(new_pop)
                front_sorted = sorted(front, key=lambda i: i.crowding_distance, reverse=True)
                new_pop.extend(front_sorted[:remaining])
                break
        return new_pop

    def _non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """
        Basic non-dominated sorting.
        """
        fronts: List[List[Individual]] = []
        S: Dict[Individual, List[Individual]] = {}
        n: Dict[Individual, int] = {}
        first_front: List[Individual] = []

        for p in population:
            S[p] = []
            n[p] = 0
            for q in population:
                if self._dominates(p, q):
                    S[p].append(q)
                elif self._dominates(q, p):
                    n[p] += 1
            if n[p] == 0:
                first_front.append(p)

        fronts.append(first_front)
        i = 0
        while fronts[i]:
            next_front: List[Individual] = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        # Remove last empty front if any
        if fronts and not fronts[-1]:
            fronts.pop()
        return fronts

    def _dominates(self, p: Individual, q: Individual) -> bool:
        """
        True if p dominates q.
        """
        assert p.objectives is not None and q.objectives is not None
        better_or_equal = all(a <= b for a, b in zip(p.objectives, q.objectives))
        strictly_better = any(a < b for a, b in zip(p.objectives, q.objectives))
        return better_or_equal and strictly_better

    def _assign_crowding_distance(self, front: List[Individual]) -> None:
        """
        Compute crowding distance for individuals in a single front.
        """
        if not front:
            return
        num_objs = len(front[0].objectives)
        for ind in front:
            ind.crowding_distance = 0.0

        for m in range(num_objs):
            front.sort(key=lambda ind: ind.objectives[m])
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")
            min_val = front[0].objectives[m]
            max_val = front[-1].objectives[m]
            if max_val == min_val:
                continue
            for i in range(1, len(front) - 1):
                prev_val = front[i - 1].objectives[m]
                next_val = front[i + 1].objectives[m]
                front[i].crowding_distance += (next_val - prev_val) / (max_val - min_val)

    def _extract_pareto_front(self, population: List[Individual]) -> List[Individual]:
        """
        Return the first non-dominated front as the Pareto set.
        """
        fronts = self._non_dominated_sort(population)
        return fronts[0] if fronts else []
    
    """ def run_nsga(self, source_code):
        # TODO: Implement NSGA-II algorithm logic
        # It could use the CandidateGenerator, Applier, and MetricCalculator
        # to find the optimal set of refactorings for the given source code
        pass"""
