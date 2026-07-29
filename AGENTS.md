# AGENTS.md

## Repository Expectations

* Only implement what is explicitly requested. Do not add extra features, abstractions, or refactors unless they are required.
* Prefer the simplest solution that satisfies the request.
* Keep changes minimal and localized. Modify as few files as possible.
* Follow the existing coding style and project structure.
* Reuse existing utilities instead of introducing duplicate implementations.
* Do not change public APIs or experiment interfaces unless explicitly requested.
* Do not introduce new dependencies unless necessary.
* If a requirement is ambiguous, ask for clarification instead of making assumptions.

## Research Code Guidelines

* Favor readability and iteration speed over premature abstraction.
* Avoid over-engineering. Research code is allowed to be simple and task-specific.
* Do not generalize code for hypothetical future use cases.
* Preserve the existing experimental workflow unless explicitly asked to change it.
* Keep experiment logic explicit rather than hiding it behind unnecessary abstractions.
* When implementing a new idea, minimize changes outside the relevant experiment.
* Do not silently change default hyperparameters, training behavior, evaluation metrics, or random seeds.
* Clearly separate experimental code from reusable utilities.
* If changing the behavior of a public utility, document the change in `docs/`.
* Unless explicitly requested, do not modify existing experiment results, checkpoints, or configuration files.
* Specify the input tensor shape at the function entry
* Include a docstring for important functions. If a function implements a specific mathematical formula, the corresponding formula must be documented using LaTex within the function's docstring.