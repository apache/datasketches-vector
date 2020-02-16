<!--
    Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.
-->

[![Build Status](https://travis-ci.org/apache/incubator-datasketches-vector.svg?branch=master)](https://travis-ci.org/apache/incubator-datasketches-vector)

=================

# DataSketches Vector Library (Experimental)
This code is offered "as is" and may not be to the quality of code in, for example, the core datasketches-java repository. 


## Documentation

### [DataSketches Library Website](https://datasketches.apache.org/)


## Build Instructions

### JDK8 is required to compile
This DataSketches component is pure Java and you must compile using JDK 8.

### Recommended Build Tool
This DataSketches component is structured as a Maven project and Maven is the recommended Build Tool.

There are two types of tests: normal unit tests and tests run by the strict profile.  

To run normal unit tests:

    $ mvn clean test

To run the strict profile tests:

    $ mvn clean test -P strict

To install jars built from the downloaded source:

    $ mvn clean install -DskipTests=true

This will create the following jars:

* datasketches-vector-X.Y.Z-incubating.jar The compiled main class files.
* datasketches-vector-X.Y.Z-incubating-tests.jar The compiled test class files.
* datasketches-vector-X.Y.Z-incubating-sources.jar The main source files.
* datasketches-vector-X.Y.Z-incubating-test-sources.jar The test source files
* datasketches-vector-X.Y.Z-incubating-javadoc.jar  The compressed Javadocs.

### Dependencies

#### Run-time
There are two run-time dependencies:

* org.ojalgo : ojalgo 
* org.apache.datasketches : datasketches-memory

#### Testing
See the pom.xml file for test dependencies.

## How to Contact Us
* We have two ASF [the-ASF.slack.com](http://the-ASF.slack.com) slack channels:
    * datasketches -- general user questions
    * datasketches-dev -- similar to our Apache [Developers Mail list](https://lists.apache.org/list.html?dev@datasketches.apache.org), except more interactive, but not as easily searchable.

* For bugs and performance issues: [Issues for datasketches-vector](https://github.com/apache/incubator-datasketches-vector/issues) 

* For general questions about using the library please subscribe: [Users Mail List](https://lists.apache.org/list.html?users@datasketches.apache.org)

* If you are interested in contributing please subscribe: [Developers Mail list](https://lists.apache.org/list.html?dev@datasketches.apache.org)
