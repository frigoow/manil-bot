from learn.run_dmc import (benchmark_opponents, config_changes,
                           generation_exploration_args, generation_league,
                           best_response_error, trainer_learner_args,
                           trainer_lr_args, is_full_eval_cycle)


def test_normal_cycle_uses_random_and_direct_anchor():
    assert benchmark_opponents("q:incumbent.pt", 1000, 1200, False, 400) == [
        ("random", 1000),
        ("q:incumbent.pt", 1200),
    ]


def test_full_ladder_keeps_distinct_direct_anchor():
    opponents = benchmark_opponents(
        "q:incumbent.pt", 1000, 1200, True, 400)
    assert opponents == [
        ("random", 2000),
        ("ismcts:100", 400),
        ("ismcts:500", 400),
        ("ismcts:1000", 400),
        ("ismcts:5000", 400),
        ("q:incumbent.pt", 1200),
    ]


def test_full_ladder_does_not_duplicate_ismcts_anchor():
    opponents = benchmark_opponents("ismcts:500", 1000, 1200, True, 400)
    assert opponents.count(("ismcts:500", 400)) == 1
    assert not any(opp == "ismcts:500" and deals == 1200
                   for opp, deals in opponents)


def test_resume_config_changes_are_explicit():
    assert config_changes(
        {"anchor": "q:old.pt", "steps": 1500},
        {"anchor": "q:new.pt", "steps": 1500, "augment": True},
    ) == {
        "anchor": {"from": "q:old.pt", "to": "q:new.pt"},
        "augment": {"from": None, "to": True},
    }


def test_generation_league_can_pin_q_anchor_while_rolling_recent():
    league = [f"q{i}.pt" for i in range(12)]
    assert generation_league(
        league, "q:champ.pt", 4, retain_anchor=True) == [
            "champ.pt", "q9.pt", "q10.pt", "q11.pt"]


def test_generation_league_deduplicates_anchor_and_ignores_non_q_anchor():
    league = ["champ.pt", "q1.pt", "q2.pt", "q3.pt"]
    assert generation_league(
        league, "q:champ.pt", 3, retain_anchor=True) == [
            "champ.pt", "q2.pt", "q3.pt"]
    assert generation_league(
        league, "ismcts:500", 3, retain_anchor=True) == league[-3:]
    assert generation_league(
        league, "q:champ.pt", 1, retain_anchor=True) == ["champ.pt"]


def test_generation_forwards_explicit_epsilon_ladder():
    assert generation_exploration_args(None) == []
    assert generation_exploration_args("0.01") == ["--eps-ladder", "0.01"]


def test_best_response_league_is_exactly_the_frozen_target():
    """--league-size 1 --retain-anchor-in-league pins self-play to the
    anchor however many of the run's own checkpoints state['league'] holds
    (run_dmc keeps appending them)."""
    own = [f"runs/br/q_cycle{i:03d}.pt" for i in range(15)]
    assert generation_league(own, "q:bp.pt", 1, retain_anchor=True) == [
        "bp.pt"]


def test_best_response_requires_pinned_full_league():
    ok = dict(learner_team=0, league_frac=1.0, anchor="q:bp.pt",
              league_size=1, retain_anchor=True)
    assert best_response_error(**ok) is None
    assert best_response_error(**{**ok, "learner_team": None,
                                  "league_frac": 0.2}) is None
    for bad in ({"league_frac": 0.2}, {"anchor": "ismcts:500"},
                {"league_size": 2}, {"retain_anchor": False},
                {"learner_team": 1}):
        assert best_response_error(**{**ok, **bad}), bad


def test_run_dmc_forwards_learner_team():
    import inspect

    import learn.run_dmc as R

    assert trainer_learner_args(None) == []
    assert trainer_learner_args(0) == ["--learner-team", "0"]
    assert "trainer_learner_args(args.learner_team)" in inspect.getsource(
        R.main)


def test_run_dmc_forwards_learning_rate():
    """Unset keeps the trainer defaults (existing runs unchanged); set
    values reach the trainer command and round-trip exactly."""
    import inspect

    import learn.run_dmc as R

    assert trainer_lr_args(None, None) == []
    args = trainer_lr_args(1e-4, 3e-5)
    assert args == ["--lr", "0.0001", "--lr-final", "3e-05"]
    assert float(args[1]) == 1e-4 and float(args[3]) == 3e-5
    assert "trainer_lr_args(args.lr, args.lr_final)" in inspect.getsource(
        R.main)


def test_full_eval_cadence_and_never():
    assert [is_full_eval_cycle(c, 5) for c in range(6)] == [
        True, False, False, False, True, False]
    assert not any(is_full_eval_cycle(c, 0) for c in range(20))
