from dsr.train import sampler


def test_sampler_registry():
    predictor_cls = sampler.get_predictor("analytic")
    assert predictor_cls is not None
