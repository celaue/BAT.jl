# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct AHMIntegration <: IntegrationAlgorithm

Adaptive Harmonic Mean Integration algorithm
([Caldwell et al.](https://arxiv.org/abs/1808.08051)).

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct AHMIntegration{
    WA<:WhiteningAlgorithm,
    AC<:AutocorLenAlgorithm
} <: IntegrationAlgorithm
    whitening::WA = CholeskyPartialWhitening()
    autocorlen::AC = GeyerAutocorLen()
    volumetype::Symbol = :HyperRectangle
    max_startingIDs::Int = 10000
    max_startingIDs_fraction::Float64 = 2.5
    rect_increase::Float64 = 0.1
    warning_minstartingids::Int = 16
    dotrimming::Bool = true

    """
    List of uncertainty estimation methods
    to use, first entry will be used for primary result. Valid values:

    * `:cov`: Integral uncertainty for integration regions is estimated based
      on covariance of integrals of subsets of samples in the regions
  
    * `:ess`: Integral uncertainty for integration regions is estimated based
      on estimated effective number of samples in each region.
    """
    uncertainty::Vector{Symbol} = [:cov]
end
export AHMIntegration


function bat_integrate_impl(target::DensitySampleVector, algorithm::AHMIntegration)
    hmi_data = HMIData(unshaped.(target))

    integrationvol = algorithm.volumetype

    uncertainty_est_mapping = Dict(
        :cov => ("cov_weighted" => hm_combineresults_covweighted!),
        :ess => ("ess_weighted" => hm_combineresults_analyticestimation!),
    )

    uncertainty_estimators = Dict(uncertainty_est_mapping[u] for u in algorithm.uncertainty)

    primary_uncertainty_estimator = uncertainty_est_mapping[first(algorithm.uncertainty)][1]

    hmi_settings = HMISettings(
        _amhi_whitening_func(algorithm.whitening),
        algorithm.max_startingIDs,
        algorithm.max_startingIDs_fraction,
        algorithm.rect_increase,
        true,
        algorithm.warning_minstartingids,
        algorithm.dotrimming,
        uncertainty_estimators
    )

    hm_integrate!(hmi_data, integrationvol, settings = hmi_settings)

    result = hmi_data.integralestimates[primary_uncertainty_estimator].final

    integral = Measurements.measurement(result.estimate, result.uncertainty)
    (result = integral, info = hmi_data) # this is needed to debug integration. can be removed later.
end


function bat_integrate_impl(target::AnySampleable, algorithm::AHMIntegration)
    npar = totalndof(varshape(target))
    samples = bat_sample(target).result::DensitySampleVector
    bat_integrate(samples, algorithm)
end

function bat_integrate_impl(target::SampledDensity, algorithm::AHMIntegration)
    bat_integrate_impl(target.samples, algorithm)
end
