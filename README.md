# 🪑 AI-Powered Spatial Occupancy Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/AI-YOLOv11-orange)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

> **Real-time computer vision system for tracking, analyzing, and reporting space utilization metrics.**

---

## 📸 Demo
![Demo GIF](docs/demo.gif)
*(Note: Replace this with a screen recording of your system in action!)*

## 🚀 Overview
This project is an end-to-end analytics platform that converts raw CCTV footage into actionable business insights. Unlike simple object detectors, this system maintains **persistent identity** of furniture and people, allowing it to calculate advanced metrics like "dwell time," "turnover rate," and "peak usage windows."

It solves common computer vision challenges such as **occlusion**, **id-switching**, and **multi-camera overlap** through custom state-management logic.

## ✨ Key Features
* **Advanced Tracking:** Utilizes **YOLOv11** + **DeepSort** for robust ID tracking even during temporary occlusions.
* **Drift Compensation:** Implements a weighted moving average algorithm to stabilize bounding boxes and prevent "ghost" detections.
* **Multi-Camera Support:** Features a custom "Hysteresis Manager" to resolve conflicts when two cameras view the same object.
* **Analytics Dashboard:** A **FastAPI** backend serves a web interface for visualizing real-time occupancy graphs and historical ledgers.
* **Optimized Performance:** Runs at **30+ FPS** on consumer hardware using CUDA acceleration and batch processing.

## 🏗️ System Architecture
```mermaid
graph TD
    A[Video Source] --> B(YOLOv11 Detection)
    B --> C(DeepSort Tracking)
    C --> D{Logic Engine}
    D -->|Match IDs| E[State Manager]
    E -->|Write| F[(SQLite Database)]
    F --> G[FastAPI Backend]
    G --> H[Web Dashboard]