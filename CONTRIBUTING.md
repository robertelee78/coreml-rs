# Contributing to coreml-native

Thank you for your interest in contributing. This document covers the essentials.

## Reporting Bugs and Requesting Features

Open an issue on GitHub at <https://github.com/robertelee78/coreml-native/issues>. For bugs, include:

- The Rust toolchain version (`rustc --version`)
- macOS version and hardware (Intel vs Apple Silicon)
- A minimal reproduction case
- The full error output

## Development Setup

```bash
git clone https://github.com/robertelee78/coreml-native.git
cd coreml-native
cargo build
cargo test
```

**Minimum supported Rust version (MSRV):** 1.75+

### Integration Tests

Integration tests require macOS with a CoreML model fixture placed in the expected
test directory. They will be skipped automatically on non-macOS platforms. See the
`tests/` directory for fixture expectations.

## Code Style

Before submitting, please run:

```bash
cargo fmt --all
cargo clippy --all-targets -- -D warnings
```

All code must compile without warnings and pass `clippy` checks.

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make your changes with clear, descriptive commits.
3. Ensure `cargo test`, `cargo fmt --check`, and `cargo clippy` all pass.
4. Open a pull request against `main` with a summary of what changed and why.
5. A maintainer will review your PR. Please respond to feedback promptly.

## License

By contributing, you agree that your contributions will be licensed under the same
terms as the project: Apache-2.0 OR MIT, at the user's option.
