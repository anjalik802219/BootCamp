#!/usr/bin/env python
# coding: utf-8

# # BootCamp 1st Day Assignment 
# Anjali Kumari(2246810)

# # 1. Convert a string to a zigzag pattern on a given number of rows and then read it row by row.Input: s = "PAYPALISHIRING", numRows = 3

# In[2]:


def convert_to_zigzag(s: str, numRows: int) -> str:
    if numRows == 1 or numRows >= len(s):
        return s
    
    # Create a list of empty strings for each row
    rows = [''] * numRows
    cur_row = 0
    going_down = False
    
    # Traverse the string and fill the rows
    for char in s:
        rows[cur_row] += char
        
        # Determine whether to go down or up
        if cur_row == 0 or cur_row == numRows - 1:
            going_down = not going_down
        
        cur_row += 1 if going_down else -1
    
    # Combine all the rows into one string
    return ''.join(rows)

# Example usage:
s = "PAYPALISHIRING"
numRows = 3
result = convert_to_zigzag(s, numRows)
print(result)


# # 2. Implement a method to perform basic string compression using the counts of repeated characters.Input: "aabcccccaaa" 

# In[4]:


def compress_string(s: str) -> str:
    if not s:
        return ""
    
    compressed = []
    count = 1
    
    # Iterate through the string
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            # Increment the count if the current char is the same as the previous one
            count += 1
        else:
            # Append the previous character and its count to the result
            compressed.append(s[i - 1] + str(count))
            count = 1
    
    # Append the last character and its count
    compressed.append(s[-1] + str(count))
    
    # Join the list into a string
    compressed_str = ''.join(compressed)
    
    # Return the original string if the compressed one is not shorter
    return compressed_str if len(compressed_str) < len(s) else s

# Example usage:
s = "aabcccccaaa"
result = compress_string(s)
print(result)


# # 3. Write a function that generates all possible permutations of a given string.Input: "ABC" 

# In[5]:


def generate_permutations(s: str) -> list:
    # Base case: if the string is empty, return an empty list
    if len(s) == 0:
        return []
    
    # Base case: if the string has only one character, return the character itself
    if len(s) == 1:
        return [s]
    
    # List to store all permutations
    perms = []
    
    # Iterate through the string and recursively generate permutations
    for i in range(len(s)):
        # Current character to be fixed
        current_char = s[i]
        
        # Remaining characters
        remaining_chars = s[:i] + s[i+1:]
        
        # Recursively generate all permutations of the remaining characters
        for perm in generate_permutations(remaining_chars):
            perms.append(current_char + perm)
    
    return perms

# Example usage:
s = "ABC"
result = generate_permutations(s)
print(result)


# In[6]:


import itertools

def generate_permutations_itertools(s: str) -> list:
    return [''.join(p) for p in itertools.permutations(s)]

# Example usage:
s = "ABC"
result = generate_permutations_itertools(s)
print(result)


# # 4. Given an array of strings, group anagrams together.Input: ["eat", "tea", "tan", "ate", "nat", "bat"] 

# In[7]:


from collections import defaultdict

def group_anagrams(strs: list) -> list:
    anagrams = defaultdict(list)
    
    # Iterate through each word in the list
    for word in strs:
        # Sort the characters in the word and use it as the key
        sorted_word = ''.join(sorted(word))
        
        # Append the word to the list corresponding to the sorted key
        anagrams[sorted_word].append(word)
    
    # Return the grouped anagrams as a list of lists
    return list(anagrams.values())

# Example usage:
strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
result = group_anagrams(strs)
print(result)


# # 5. Write a function to multiply two large numbers represented as strings.Input: num1 = "123", num2 = "456" 

# In[8]:


def multiply_strings(num1: str, num2: str) -> str:
    # Edge case: if either number is "0", the result is "0"
    if num1 == "0" or num2 == "0":
        return "0"
    
    # Initialize a result list with zeros, with a maximum possible length of num1 + num2
    result = [0] * (len(num1) + len(num2))
    
    # Reverse the strings to handle the multiplication from least significant digit to most significant
    num1 = num1[::-1]
    num2 = num2[::-1]
    
    # Multiply each digit of num1 with each digit of num2
    for i in range(len(num1)):
        for j in range(len(num2)):
            # Multiply the digits and add the product to the correct position in the result
            product = int(num1[i]) * int(num2[j])
            result[i + j] += product
            
            # Handle carry over
            result[i + j + 1] += result[i + j] // 10
            result[i + j] %= 10
    
    # Reverse the result list and convert it back to a string, skipping any leading zeros
    result = result[::-1]
    
    # Skip leading zeros (if any)
    i = 0
    while i < len(result) and result[i] == 0:
        i += 1
    
    # Convert the digits back to a string
    return ''.join(map(str, result[i:]))

# Example usage:
num1 = "123"
num2 = "456"
result = multiply_strings(num1, num2)
print(result)


# # 6. Given two strings, check if one is a rotation of the other using only one call to a string method.Input: str1 = "ABCD", str2 = "DABC" 

# In[9]:


def is_rotation(str1: str, str2: str) -> bool:
    # Check if the lengths of the two strings are the same, if not, they can't be rotations
    if len(str1) != len(str2):
        return False
    
    # Check if str2 is a substring of str1 + str1
    return str2 in (str1 + str1)

# Example usage:
str1 = "ABCD"
str2 = "DABC"
result = is_rotation(str1, str2)
print(result)


# # 7. Given a string containing just the characters (, ), {, }, [, and ], determine if the input string is valid. Input: "()[]{}"

# In[10]:


def is_valid_parentheses(s: str) -> bool:
    # Dictionary to hold matching pairs
    matching_bracket = {')': '(', '}': '{', ']': '['}
    # Stack to hold opening brackets
    stack = []
    
    for char in s:
        if char in matching_bracket.values():
            # If it's an opening bracket, push it onto the stack
            stack.append(char)
        elif char in matching_bracket.keys():
            # If it's a closing bracket, check if the stack is empty or if it matches
            if not stack or stack.pop() != matching_bracket[char]:
                return False
    
    # If the stack is empty, all brackets were matched correctly
    return len(stack) == 0

# Example usage:
s = "()[]{}"
result = is_valid_parentheses(s)
print(result)


# # 8. Implement the function atoi which converts a string to an integer.Input: "4193 with words"

# In[11]:


def my_atoi(s: str) -> int:
    # Define constants for the integer limits
    INT_MAX = 2**31 - 1
    INT_MIN = -2**31
    
    # Remove leading whitespaces
    s = s.lstrip()
    
    if not s:
        return 0
    
    # Initialize variables
    sign = 1  # Assume positive by default
    result = 0
    index = 0
    n = len(s)
    
    # Check for sign
    if s[index] == '-':
        sign = -1
        index += 1
    elif s[index] == '+':
        index += 1
    
    # Convert digits to integer
    while index < n and s[index].isdigit():
        digit = int(s[index])
        
        # Check for overflow before adding the digit
        if result > (INT_MAX - digit) // 10:
            return INT_MAX if sign == 1 else INT_MIN
        
        result = result * 10 + digit
        index += 1
    
    return sign * result

# Example usage:
s = "4193 with words"
result = my_atoi(s)
print(result)


# # 9. Write a function that generates the nth term of the "count and say" sequence.Input: n = 4 

# In[12]:


def count_and_say(n: int) -> str:
    # Base case
    if n == 1:
        return "1"
    
    # Start with the first term
    term = "1"
    
    # Generate terms from 2 to n
    for _ in range(1, n):
        next_term = ""
        count = 1
        
        # Iterate through the current term to build the next term
        for i in range(1, len(term)):
            if term[i] == term[i - 1]:
                count += 1  # Increase count if the same digit is found
            else:
                # Append count and digit to next term
                next_term += str(count) + term[i - 1]
                count = 1  # Reset count for the new digit
        
        # Append the last counted digit
        next_term += str(count) + term[-1]
        
        # Move to the next term
        term = next_term
    
    return term

# Example usage:
n = 4
result = count_and_say(n)
print(result)


# # 10. Given two strings s and t, return the minimum window in s which will contain all the characters in t. Input: s = "ADOBECODEBANC", t = "ABC"

# In[13]:


from collections import Counter, defaultdict

def min_window(s: str, t: str) -> str:
    if not t or not s:
        return ""

    # Dictionary to count characters in t
    dict_t = Counter(t)
    
    # Number of unique characters in t that must be present in the window
    required = len(dict_t)
    
    # Left and right pointers for the sliding window
    l, r = 0, 0
    
    # Formed is to keep track of how many unique characters in t are currently in the window
    formed = 0
    
    # Dictionary to keep count of characters in the current window
    window_counts = defaultdict(int)
    
    # Result tuple (window length, left, right)
    ans = float("inf"), None, None
    
    while r < len(s):
        # Add the character from the right end of the window
        character = s[r]
        window_counts[character] += 1
        
        # Check if the current character's count matches the required count in t
        if character in dict_t and window_counts[character] == dict_t[character]:
            formed += 1
        
        # Try to contract the window until it's no longer valid
        while l <= r and formed == required:
            character = s[l]
            
            # Save the smallest window and its indices
            if r - l + 1 < ans[0]:
                ans = (r - l + 1, l, r)
            
            # Remove the leftmost character from the window
            window_counts[character] -= 1
            if character in dict_t and window_counts[character] < dict_t[character]:
                formed -= 1
            
            # Move the left pointer to the right
            l += 1
        
        # Expand the window by moving the right pointer to the right
        r += 1
    
    # Return the minimum window or an empty string if no valid window was found
    return "" if ans[0] == float("inf") else s[ans[1]: ans[2] + 1]

# Example usage:
s = "ADOBECODEBANC"
t = "ABC"
result = min_window(s, t)
print(result)


# # 11. Given a string, find the length of the longest substring without repeating characters. Input: "abcabcbb"

# In[14]:


def length_of_longest_substring(s: str) -> int:
    char_set = set()  # Set to store characters in the current window
    left = 0  # Left pointer for the sliding window
    max_length = 0  # Maximum length of substring without repeating characters

    for right in range(len(s)):
        # If the character is already in the set, move the left pointer to remove duplicates
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        
        # Add the current character to the set
        char_set.add(s[right])
        
        # Update the maximum length
        max_length = max(max_length, right - left + 1)

    return max_length

# Example usage:
s = "abcabcbb"
result = length_of_longest_substring(s)
print(result)


# # 12. Given three strings s1, s2, and s3, determine if s3 is formed by the interleaving of s1 and s2.Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac" 

# In[15]:


def is_interleave(s1: str, s2: str, s3: str) -> bool:
    # Lengths of the strings
    len1, len2, len3 = len(s1), len(s2), len(s3)
    
    # Check if the lengths match
    if len1 + len2 != len3:
        return False
    
    # Create a 2D DP array
    dp = [[False] * (len2 + 1) for _ in range(len1 + 1)]
    
    # Initialize the DP array
    dp[0][0] = True  # Empty strings can form an empty string
    
    # Fill the first row
    for j in range(1, len2 + 1):
        dp[0][j] = dp[0][j - 1] and s2[j - 1] == s3[j - 1]
    
    # Fill the first column
    for i in range(1, len1 + 1):
        dp[i][0] = dp[i - 1][0] and s1[i - 1] == s3[i - 1]
    
    # Fill the rest of the DP table
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            dp[i][j] = (dp[i - 1][j] and s1[i - 1] == s3[i + j - 1]) or                        (dp[i][j - 1] and s2[j - 1] == s3[i + j - 1])
    
    # The answer is in the bottom-right corner of the DP table
    return dp[len1][len2]

# Example usage:
s1 = "aabcc"
s2 = "dbbca"
s3 = "aadbbcbcac"
result = is_interleave(s1, s2, s3)
print(result)


# # 13. Write a function to convert a Roman numeral to an integer. Input: "MCMXCIV"

# In[16]:


def roman_to_int(s: str) -> int:
    # Map of Roman numerals to integers
    roman_values = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000
    }
    
    total = 0
    prev_value = 0
    
    # Traverse the Roman numeral from right to left
    for char in reversed(s):
        value = roman_values[char]
        
        # If the current value is less than the previous value, subtract it
        if value < prev_value:
            total -= value
        else:
            total += value
        
        # Update the previous value to the current value
        prev_value = value
    
    return total

# Example usage:
s = "MCMXCIV"
result = roman_to_int(s)
print(result)


# # 14. Find the longest common substring between two strings. Input: str1 = "ABABC", str2 = "BABCA"

# In[17]:


def longest_common_substring(str1: str, str2: str) -> str:
    len1, len2 = len(str1), len(str2)
    
    # Create a 2D array to store lengths of longest common suffixes
    # Initialize with 0 for all pairs
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    max_length = 0  # Length of the longest common substring
    end_index = 0  # Ending index of the longest common substring in str1
    
    # Fill the dp array
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_index = i  # Update the end index of the longest common substring
            else:
                dp[i][j] = 0  # Reset length if characters do not match

    # Extract the longest common substring using the end index and max length
    longest_substring = str1[end_index - max_length:end_index]
    
    return longest_substring

# Example usage:
str1 = "ABABC"
str2 = "BABCA"
result = longest_common_substring(str1, str2)
print(result)


# # 15. Given a string s and a dictionary of words, check if s can be segmented into a space-separated sequence of one or more dictionary words.Input: s = "leetcode", wordDict = ["leet", "code"] 

# In[18]:


def word_break(s: str, wordDict: list[str]) -> bool:
    word_set = set(wordDict)  # Convert list to set for faster lookup
    dp = [False] * (len(s) + 1)  # DP array
    dp[0] = True  # Base case: empty string can be segmented

    # Iterate through the string
    for i in range(1, len(s) + 1):
        for j in range(i):
            # Check if the substring s[j:i] is in the word set
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break  # No need to check further for this i

    return dp[len(s)]  # The result for the entire string

# Example usage:
s = "leetcode"
wordDict = ["leet", "code"]
result = word_break(s, wordDict)
print(result)


# # 16.Remove the minimum number of invalid parentheses to make the input string valid. Return all possible results. Input: "()())()"

# In[19]:


from collections import deque

def is_valid(s: str) -> bool:
    # Helper function to check if a string has valid parentheses
    count = 0
    for char in s:
        if char == '(':
            count += 1
        elif char == ')':
            count -= 1
        if count < 0:
            return False
    return count == 0

def remove_invalid_parentheses(s: str) -> list[str]:
    # BFS initialization
    queue = deque([s])
    visited = set([s])
    found = False
    result = []
    
    while queue:
        current = queue.popleft()
        
        if is_valid(current):
            result.append(current)
            found = True
        
        if found:
            continue
        
        # Generate all possible strings by removing one parenthesis at a time
        for i in range(len(current)):
            if current[i] not in ('(', ')'):
                continue
            new_str = current[:i] + current[i+1:]
            if new_str not in visited:
                visited.add(new_str)
                queue.append(new_str)
    
    return result

# Example usage:
s = "()())()"
result = remove_invalid_parentheses(s)
print(result)


# In[ ]:




