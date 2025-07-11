# Introduction

Welcome to the documentation for the Dies Behavior Tree (BT) and Rhai Scripting system.

## What is this system?

This system combines the power of **Behavior Trees**, a formal and robust way to model agent behavior, with the flexibility of the **Rhai scripting language**. This allows for creating complex, dynamic, and even hot-reloadable behaviors for our robots in a declarative and easy-to-understand way.

The system features a **dynamic role assignment engine** that automatically adapts team strategy based on game state and player availability. Instead of hard-coding static behaviors, you define flexible roles that are assigned intelligently based on current conditions.

Instead of hard-coding behaviors in Rust, we can define them in simple Rhai scripts, which are then executed by the game executor.

## Key Features

- **Dynamic Role Assignment**: Roles are assigned automatically based on game state, player count, and scoring functions
- **Real-time Adaptation**: Strategy adapts immediately when players join/leave or game state changes
- **Intelligent Coordination**: Built-in constraint solving ensures proper team formation
- **Hot Reloading**: Strategy changes take effect without recompiling or restarting
- **Modular Design**: Reusable behavior trees and scoring functions

## Who is this guide for?

This guide is intended for developers working on the Dies project, specifically those involved in creating and defining robot strategies and behaviors. A basic understanding of Rust is assumed. While prior knowledge of Behavior Trees and Rhai is helpful, this guide provides introductory material for both.

## How is this guide structured?

This guide is divided into the following sections:

- **Introduction to Behavior Trees**: A primer on the concepts of Behavior Trees.
- **Scripting with Rhai**: An introduction to the Rhai scripting language and its use in our system.
- **Getting Started**: A tutorial to write your first dynamic strategy script.
- **Role Assignment API**: Complete reference for the dynamic role assignment system.
- **API Reference**: A detailed reference of all the available nodes, skills, and helper functions that can be used in scripts.
