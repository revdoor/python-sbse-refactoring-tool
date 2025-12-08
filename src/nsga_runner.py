"""
This module defines the NSGARunner class,
which is used to run the NSGA-II algorithm
to find the optimal set of refactorings
for the given source code.
"""
from __future__ import annotations
from collections.abc import Sequence
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
import random

from candidate_generator import CandidateGenerator
from refactoring_operator import RefactoringOperator
from metric_calculator import MetricCalculator
from applier import Applier


@dataclass
class RefactoringPlan:
    """
    A candidate solution: a sequence of refactoring operators.
    """
    genes: List[RefactoringOperator]

    def __hash__(self):
        return hash(tuple(self.genes))


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

    def __hash__(self):
        return hash(self.plan)


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
            source_code: str,
            candidate_generator: Callable[[], RefactoringPlan],
            pop_size: int = 10,
            n_generations: int = 30,
            cx_prob: float = 0.7,
            mut_prob: float = 0.3,
            random_seed: int = 0,
            candidate_pool: Optional[Sequence[RefactoringOperator]] = None,
    ):
        self.source_code = source_code
        self.candidate_generator = candidate_generator
        self.applier = Applier()
        self.metric_calculator = MetricCalculator()

        self.pop_size = pop_size
        self.n_generations = n_generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.random = random.Random(random_seed)

        self.candidate_pool: List[RefactoringOperator] = list(candidate_pool) if candidate_pool else []

    @classmethod
    def from_source_code(
            cls,
            source_code: str,
            pop_size: int = 20,
            n_generations: int = 30,
            cx_prob: float = 0.7,
            mut_prob: float = 0.3,
            random_seed: int = 0,
    ) -> "NSGARunner":
        """
        Helper to build an NSGARunner directly from raw source code.
        It will:
          - ask CandidateGenerator for all possible RefactoringOperator candidates
          - build a candidate_generator() that samples random subsets of them
        """
        all_candidates: Sequence[RefactoringOperator] = CandidateGenerator.generate_candidates(source_code)
        rnd = random.Random(random_seed)

        def make_random_plan() -> RefactoringPlan:
            """
            Sample a random subset of the global candidate list.

            For now:
            - choose a random length in [1, min(5, len(all_candidates))]
            - sample without replacement
            """
            if not all_candidates:
                return RefactoringPlan(genes=[])

            max_len = min(5, len(all_candidates))
            length = rnd.randint(1, max_len)
            genes = rnd.sample(list(all_candidates), length)
            return RefactoringPlan(genes=genes)

        return cls(
            source_code=source_code,
            candidate_generator=make_random_plan,
            pop_size=pop_size,
            n_generations=n_generations,
            cx_prob=cx_prob,
            mut_prob=mut_prob,
            random_seed=random_seed,
            candidate_pool=all_candidates,
        )

    def run(self) -> List[Individual]:
        """
        Main entry point: run NSGA-II and return the final Pareto front.
        """
        population = self._init_population()
        self._evaluate_population(population)

        for gen in range(self.n_generations):
            print(f"[NSGA-II] Generation {gen + 1}/{self.n_generations}")

            offspring = self._make_offspring(population)
            self._evaluate_population(offspring)

            population = self._environmental_selection(population + offspring)

            self._print_generation_best(gen + 1, population)

        pareto_front = self._extract_pareto_front(population)

        self._print_final_results(pareto_front)

        return pareto_front

    def _print_final_results(self, pareto_front: List[Individual]) -> None:
        if not pareto_front:
            print("\n[Final Results] No solutions found in Pareto front.")
            return

        print("\n" + "=" * 80)
        print("FINAL OPTIMIZATION RESULTS")
        print("=" * 80)
        print(f"Pareto Front Size: {len(pareto_front)}")
        print("=" * 80)

        best_by_metric = {
            "Structural": min(pareto_front, key=lambda ind: ind.objectives[0] if ind.objectives else float('inf')),
            "Cost": min(pareto_front, key=lambda ind: ind.objectives[1] if ind.objectives else float('inf')),
            "Readability": min(pareto_front, key=lambda ind: ind.objectives[2] if ind.objectives else float('inf')),
        }

        printed_plans = set()

        for metric_name, ind in best_by_metric.items():
            plan_hash = hash(ind.plan)

            print(f"\n{'─' * 80}")
            print(f"🏆 BEST BY {metric_name.upper()} SCORE")
            print(f"{'─' * 80}")

            self._print_individual_detailed(ind)

            if plan_hash in printed_plans:
                print(f"\n[Code] (Same as above - already printed)")
            else:
                printed_plans.add(plan_hash)

                transformed_code = self._apply_plan(ind.plan)
                print(f"\n[Transformed Code]")
                print("```python")
                print(transformed_code)
                print("```")

        print(f"\n{'=' * 80}")
        print("COMPARISON SUMMARY")
        print("=" * 80)
        self._print_comparison_summary(pareto_front)
        print("=" * 80 + "\n")

    def _print_individual_detailed(self, ind: Individual, indent: str = "") -> None:
        if ind.objectives is None:
            print(f"{indent}(not evaluated)")
            return

        structural, cost, neg_readability = ind.objectives
        readability = -neg_readability  # 원래 값으로 복원

        print(f"{indent}📊 Metrics:")
        print(f"{indent}   • Structural Score: {structural:.4f}")
        print(f"{indent}   • Cost Score: {cost:.4f}")
        print(f"{indent}   • Readability Score: {readability:.4f}")
        print(f"{indent}   • Number of Refactorings: {len(ind.plan.genes)}")

        if ind.plan.genes:
            print(f"{indent}🔧 Applied Refactorings:")
            for i, gene in enumerate(ind.plan.genes, 1):
                print(f"{indent}   {i}. {gene}")
        else:
            print(f"{indent}🔧 Applied Refactorings: None")

    def _print_comparison_summary(self, pareto_front: List[Individual]) -> None:
        original_metrics = self.metric_calculator.calculate_metric(self.source_code)
        orig_cc, orig_sloc, orig_fan_in, orig_readability = original_metrics
        orig_structural = self._structural_objective(orig_cc, orig_sloc)
        orig_cost = self._cost_objective(orig_fan_in, RefactoringPlan(genes=[]))

        print(f"\n📋 Original Code Metrics:")
        print(f"   • Structural Score: {orig_structural:.4f}")
        print(f"   • Cost Score: {orig_cost:.4f}")
        print(f"   • Readability Score: {orig_readability:.4f}")

        if pareto_front:
            structural_scores = [ind.objectives[0] for ind in pareto_front if ind.objectives]
            cost_scores = [ind.objectives[1] for ind in pareto_front if ind.objectives]
            readability_scores = [-ind.objectives[2] for ind in pareto_front if ind.objectives]

            print(f"\n📈 Pareto Front Ranges:")
            print(f"   • Structural Score: {min(structural_scores):.4f} ~ {max(structural_scores):.4f}")
            print(f"   • Cost Score: {min(cost_scores):.4f} ~ {max(cost_scores):.4f}")
            print(f"   • Readability Score: {min(readability_scores):.4f} ~ {max(readability_scores):.4f}")

            best_structural = min(structural_scores)
            best_readability = max(readability_scores)

            if orig_structural > 0:
                structural_improvement = (orig_structural - best_structural) / orig_structural * 100
                print(f"\n✨ Best Improvements:")
                print(
                    f"   • Structural: {structural_improvement:+.1f}% {'(improved)' if structural_improvement > 0 else '(worsened)'}")

            if orig_readability > 0:
                readability_improvement = (best_readability - orig_readability) / orig_readability * 100
                print(
                    f"   • Readability: {readability_improvement:+.1f}% {'(improved)' if readability_improvement > 0 else '(worsened)'}")

    def _print_generation_best(self, generation: int, population: List[Individual]) -> None:
        """
        현재 세대의 최고 결과물을 메트릭과 함께 출력합니다.
        """
        if not population:
            return

        # Pareto front 추출
        fronts = self._non_dominated_sort(population)
        pareto_front = fronts[0] if fronts else []

        # 각 목적함수별 최적 개체 찾기
        best_structural = min(population, key=lambda ind: ind.objectives[0] if ind.objectives else float('inf'))
        best_cost = min(population, key=lambda ind: ind.objectives[1] if ind.objectives else float('inf'))
        best_readability = min(population, key=lambda ind: ind.objectives[2] if ind.objectives else float('inf'))

        print(f"\n{'=' * 60}")
        print(f"[Generation {generation}] Results")
        print(f"{'=' * 60}")
        print(f"Population size: {len(population)}, Pareto front size: {len(pareto_front)}")

        print(f"\n[Best by Structural Score]")
        self._print_individual(best_structural)

        print(f"\n[Best by Cost Score]")
        self._print_individual(best_cost)

        print(f"\n[Best by Readability Score]")
        self._print_individual(best_readability)

        # Pareto front 전체 출력 (상위 3개만)
        print(f"\n[Pareto Front (top 3)]")
        for i, ind in enumerate(pareto_front[:3]):
            print(f"  #{i + 1}:")
            self._print_individual(ind, indent="    ")

        print(f"{'=' * 60}\n")

    def _print_individual(self, ind: Individual, indent: str = "  ") -> None:
        """
        개체의 정보를 출력합니다.
        """
        if ind.objectives is None:
            print(f"{indent}(not evaluated)")
            return

        structural, cost, neg_readability = ind.objectives
        readability = -neg_readability  # 원래 값으로 복원

        print(f"{indent}Structural Score: {structural:.4f}")
        print(f"{indent}Cost Score: {cost:.4f}")
        print(f"{indent}Readability Score: {readability:.4f}")
        print(f"{indent}Num Refactorings: {len(ind.plan.genes)}")

        # 적용된 리팩토링 연산자 목록
        if ind.plan.genes:
            gene_strs = [str(g) for g in ind.plan.genes]
            print(f"{indent}Operators: {', '.join(gene_strs)}")

    def _init_population(self) -> List[Individual]:
        """
        Ask the candidate generator to create initial plans.
        """
        pop: List[Individual] = []
        for _ in range(self.pop_size):
            plan = self.candidate_generator()
            pop.append(Individual(plan=plan))
        return pop

    def _apply_plan(self, plan: RefactoringPlan) -> str:
        """
        Apply all operators in a RefactoringPlan sequentially.
        If an operator fails, skip it and continue with the next.
        """
        code = self.source_code
        for operator in plan.genes:
            try:
                code = self.applier.apply_refactoring(code, operator)
            except (ValueError, TypeError) as e:
                # Skip failed operators
                pass
        return code

    def _evaluate_population(self, population: List[Individual]) -> None:
        """
        For each individual:
          - apply its refactoring plan to a codebase snapshot
          - compute metrics
          - compute objectives tuple (to minimize)
        """
        for ind in population:
            if ind.objectives is not None:
                continue

            transformed_code = self._apply_plan(ind.plan)

            cc, sloc, fan_in, llm_read = self.metric_calculator.calculate_metric(
                transformed_code
            )

            structural_score = self._structural_objective(cc, sloc)
            cost_score = self._cost_objective(fan_in, ind.plan)
            readability_obj = -float(llm_read)

            ind.objectives = (structural_score, cost_score, readability_obj)

    def _structural_objective(self, cc: float, sloc: float) -> float:
        """
        Combine structural metrics into a single value to minimize.

        For now:
          structural = cc + alpha * sloc
        alpha is a small weight so SLOC doesn't dominate; can be tuned later.
        """
        alpha = 0.1
        return float(cc) + alpha * float(sloc)

    def _cost_objective(self, fan_in: float, plan: RefactoringPlan) -> float:
        """
        Cost / regularization objective: penalize many refactorings and high fan-in.

        For now:
          cost = num_refactorings + beta * fan_in
        """
        beta = 0.1
        num_refactorings = len(plan.genes)
        return float(num_refactorings) + beta * float(fan_in)

    def _make_offspring(self, population: List[Individual]) -> List[Individual]:
        """
        Create offspring using tournament selection, crossover, and mutation.
        """
        offspring: List[Individual] = []

        while len(offspring) < self.pop_size:
            parent1 = self._tournament_select(population)
            parent2 = self._tournament_select(population)

            child1_plan = self._clone_plan(parent1.plan)
            child2_plan = self._clone_plan(parent2.plan)

            if self.random.random() < self.cx_prob:
                child1_plan, child2_plan = self._crossover(child1_plan, child2_plan)

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
        Mutation: either drop a gene or add a new gene from the candidate pool.
        """
        genes = list(plan.genes)
        if not genes:
            if self.candidate_pool:
                new_gene = self.random.choice(self.candidate_pool)
                genes.append(new_gene)
            return RefactoringPlan(genes=genes)

        if self.random.random() < 0.5:
            idx = self.random.randrange(len(genes))
            del genes[idx]
        else:
            if self.candidate_pool:
                new_gene = self.random.choice(self.candidate_pool)
                genes.append(new_gene)

        return RefactoringPlan(genes=genes)

    def _environmental_selection(self, population: List[Individual]) -> List[Individual]:
        """
        NSGA-II environmental selection:
          - non-dominated sorting
          - crowding distance
          - keep best pop_size individuals
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
        s: Dict[Individual, List[Individual]] = {}
        n: Dict[Individual, int] = {}
        first_front: List[Individual] = []

        for p in population:
            s[p] = []
            n[p] = 0
            for q in population:
                if self._dominates(p, q):
                    s[p].append(q)
                elif self._dominates(q, p):
                    n[p] += 1
            if n[p] == 0:
                first_front.append(p)

        fronts.append(first_front)
        i = 0
        while fronts[i]:
            next_front: List[Individual] = []
            for p in fronts[i]:
                for q in s[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

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
