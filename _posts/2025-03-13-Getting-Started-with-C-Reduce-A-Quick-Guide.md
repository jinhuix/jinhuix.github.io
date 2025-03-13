---
title: "Getting Started with C-Reduce: A Quick Guide"
date: 2025-03-13 11:46:40 +0800
categories: [Debugging]
tags: [C-reduce]
comments: true
---

In software development, simplifying code is essential for improving code quality and maintainability. **C-reduce** is a tool designed to reduce code, helping developers quickly pinpoint and fix issues. In this blog, we’ll cover the basic principles of C-reduce, installation steps, usage examples, and some advanced tips.

## **1. Introduction to C-Reduce**

C-reduce is an open-source code reduction tool based on the Delta Debugging algorithm. It can automatically minimize code to the smallest compilable and runnable version that still exhibits the bug. C-reduce supports various programming languages such as C, C++, and Java, and it can be integrated with multiple compilers and testing tools.

## **2. Installing C-Reduce**

On Ubuntu, you can install the precompiled version of C-Reduce directly using apt-get. Open a terminal and run:

```shell
sudo apt-get update
sudo apt-get install creduce
creduce --version
```

This will update your package lists, install C-Reduce along with its dependencies, and then display its version to confirm a successful installation.

## **3. Using C-Reduce**

The basic usage of C-Reduce is as follows:

```bash
creduce [options] <test command> <input file>
```

- **Test command:** A command that verifies whether the code is still faulty. A return value of 0 should indicate that the error is present.
- **Input file:** The code file you wish to minimize.

For example, in the `creduce/tests` directory, there are several test files available for learning:

- `test0.sh` is an interestingness test script that returns 0 if the reduced code still triggers the error.
- `file1.c` is the C code file that needs to be minimized.

<img src="./assets/img/post/2025-03/Creduce-1.png" alt="Creduce-1" style="zoom:75%;" />

C-Reduce will repeatedly invoke the test script, deleting or modifying code fragments to generate new versions, until it finds the smallest code snippet that still triggers the error. An example output might be:

```bash
===================== done ====================

pass statistics:
  method pass_clang_binsrch :: remove-unused-function worked 1 times and failed 1 times
  method pass_balanced :: curly-inside worked 1 times and failed 8 times
  ...
  method pass_lines :: 2 worked 77 times and failed 500 times
  method pass_lines :: 1 worked 255 times and failed 997 times

******** /home/xjh/creduce/tests/tmp_test0_w3FJy/file1.c ********

void a() {
b:
  goto b;
}
int main() {}
```

This shows that C-Reduce has reduced the code down to a minimal form that still triggers the error.

**Using C-Reduce with SQLancer/SQLite3:**
 If SQLancer generates a SQL query that causes SQLite3 to fail, you can also use C-Reduce to minimize the SQL input:

1. Save the failing SQL query as `test.sql`.

2. Create a test script `test.sh` (similar to the one above but calling SQLite3 to execute the SQL file).

3. Run:

   ```bash
   creduce ./test.sh test.sql
   ```

This process will help you automatically reduce the SQL query to the smallest failure-inducing version.

## **4. Advanced Tips**

### **4.1 Custom Reduction Strategies**

C-reduce allows you to customize reduction strategies to meet specific needs. You can write your own Python script to implement a custom strategy and use the `--strategy` option to specify it.

For example, create a custom strategy script named `my_strategy.py`, then run:

```bash
creduce --strategy my_strategy.py "gcc -c %i && gcc -o example example.c && ./example" example.c
```

### **4.2 Handling Dependencies**

For complex code involving multiple files, use the `--extra-files` option to specify additional files. This ensures that these files are correctly processed during reduction.

For example, if your code depends on multiple header files:

```bash
creduce --extra-files header1.h header2.h "gcc -c %i && gcc -o example example.c && ./example" example.c
```

### **4.3 Integrating with Testing Frameworks**

To efficiently verify code correctness, you can integrate C-reduce with a testing framework. For instance, if you use CTest as your testing tool, write a test script `test.sh` and run:

```bash
creduce "./test.sh %i" example.c
```

This allows you to automate the testing process while reducing the code.

