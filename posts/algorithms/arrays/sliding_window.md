---
layout: default
title: Two pointers
parent: Arrays
grand_parent: Algorithms
---

```python
def most_water_container(heights: List[int]) -> int:
    max_water_content = float('-inf')

    left, right = 0, len(heights) - 1
    while left < right:
        # Water content = width * (bar with the lower height of the two)
        water_content = (right - left) * min(heights[left], heights[right])
        max_water_content = max(max_water_content, water_content)

        # Which pointer to update? We will be making the width smaller with
        # the next pointer update, so the only hope for increasing the water content
        # is to increse the minimum height. Hence we should update the pointer
        # of the bar with the lower height of the two.
        if heights[left] < heights[right]:
            left += 1
        else:
            right -= 1

    return max_water_content
```

```python
def subarray_less_than_target(nums: List[int], target: int) -> int:
    leader, follower = 0, 0

    total_count = 0
    while leader < len(nums):
        # We keep on increasing the `leading` pointer until we 
        # reach the end of the input or until we reach an element which
        # is not smaller than the target.
        while (leader < len(nums)) and (nums[leader] <= target):
            leader += 1

        # Once we break out of the above while loop, we have reached 
        # a new subarray (followe:leader) where all the elements are less than target.
        curr_length = leader - follower
        if curr_length > 0:
            total_count += 1
        
        # We then repeat the process from the next element.
        leader += 1
        follower = leader
    return total_count
```