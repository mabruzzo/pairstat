========
Overview
========

This directory holds files that we are using to start reimplementing pairstat as an extension module that wraps pairstat.

The plan is to define a separate python package in here:
- this is definitely going to take some work since I don't have any experience with maturin
- plus, I don't want to break all of the existing tests yet... (pairstat-rs doesn't implement all required features quite yet)

Once we are satisfied, we will combine this with pairstat.
