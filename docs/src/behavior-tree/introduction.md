# Introduction

Welcome to the documentation for the Dies Behavior Tree (BT) and Rhai Scripting system.

## What is this system?

This system combines the power of **Behavior Trees**, a formal and robust way to model agent behavior, with the flexibility of the **Rhai scripting language**. This allows for creating complex, dynamic, and even hot-reloadable behaviors for our robots in a declarative and easy-to-understand way.

Instead of hard-coding behaviors in Rust, we can define them in simple Rhai scripts, which are then executed by the game executor.

## Who is this guide for?

This guide is intended for developers working on the Dies project, specifically those involved in creating and defining robot strategies and behaviors. A basic understanding of Rust is assumed. While prior knowledge of Behavior Trees and Rhai is helpful, this guide provides introductory material for both.

## How is this guide structured?

This guide is divided into the following sections:

- **Introduction to Behavior Trees**: A primer on the concepts of Behavior Trees.
- **Scripting with Rhai**: An introduction to the Rhai scripting language and its use in our system.
- **Getting Started**: A tutorial to write your first behavior tree script.
- **API Reference**: A detailed reference of all the available nodes, skills, and helper functions that can be used in scripts.
