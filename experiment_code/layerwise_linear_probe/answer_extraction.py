import re
import regex

def _fix_fracs(string):
    """Formats fractions in LaTeX string (e.g., \frac12 -> \frac{1}{2})."""
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    """Converts a/b format to LaTeX fraction \frac{a}{b}."""
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    """Formats square roots in LaTeX string."""
    _string = re.sub(r"\\sqrt(-?[0-9.a-zA-Z]+)", r"\\sqrt{\1}", string)
    _string = re.sub(r"\\sqrt\s+(\w+)$", r"\\sqrt{\1}", _string)
    return _string


def _fix_tan(string):
    """Formats tangent functions in LaTeX string."""
    _string = re.sub(r"\\tan(-?[0-9.a-zA-Z]+)", r"\\tan{\1}", string)
    _string = re.sub(r"\\tan\s+(\w+)$", r"\\tan{\1}", _string)
    return _string


def strip_string(string):
    """Normalizes string by removing units, special chars, and formatting."""
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    string = string.replace("\\!", "")
    # string = string.replace("\\ ", "")

    # replace \\ with \
    # string = string.replace("\\\\", "\\")
    # string = string.replace("\\\\", "\\")

    if string.startswith("\\text{") and string.endswith("}"):
        string = string.split("{", 1)[1][:-1]

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("cfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "").strip()
    string = string.replace("^\\circ", "").strip()

    string = regex.sub(r"\{(c|m)?m\}(\^(2|3))?", "", string).strip()
    string = regex.sub(r"p\.m\.$", "", string).strip()
    string = regex.sub(r"(\d)\s*t$", r"\1", string).strip()

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    # string = string.replace("\\text", "")
    string = string.replace("x\\in", "")

    # remove percentage
    string = string.replace("\\%", "%")
    string = string.replace(r"\%", "%")
    # string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    string = string.replace("\\cdot", "")

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and 
    # string = string.replace("and", "")
    string = string.replace("\\mathbf", "")
    string = string.replace("\\mathrm", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string = string.replace("'", "")
    string = string.replace("\"", "")
    
    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    # if len(string.split("=")) == 2:
    #     if len(string.split("=")[0]) <= 2:
    #         string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = _fix_tan(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    string = regex.sub(r"(\\|,|\.)+$", "", string)

    return string

def extract_boxed_answers(text):
    """Extracts content inside \boxed{...} tags."""
    answers = []
    for piece in text.split('boxed{')[1:]:
        n = 0
        for i in range(len(piece)):
            if piece[i] == '{':
                n += 1
            elif piece[i] == '}':
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == '%':
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    return answers

def extract_program_output(pred_str):
    """Extracts output from the last ```output ... ``` block."""
    if "```output" not in pred_str:
        return ""
    else:
        pred_str = pred_str.split('```output')[-1]
    if '```' in pred_str:
        pred_str = pred_str.split('```')[0]
    output = pred_str.strip()
    return output

def extract_answer(pred_str, exhaust=False):
    """
    Extracts answers from prediction string using various heuristics (boxed, final answer, etc.).
    
    Args:
        pred_str: The model's prediction string.
        exhaust: If True, returns all candidates; otherwise returns the last/best one.
    """
    pred = []
    if 'final answer is $' in pred_str and '$. I hope' in pred_str:
        tmp = pred_str.split('final answer is $', 1)[1]
        pred = [tmp.split('$. I hope', 1)[0].strip()]
    elif 'boxed' in pred_str:
        boxed_answers = extract_boxed_answers(pred_str)
        # Filter out non-numeric answers like "answer" which come from prompt templates
        # Only use boxed if we found something that looks like it could be a number
        if boxed_answers and any(ans and not ans.isalpha() for ans in boxed_answers):
            pred = boxed_answers
        elif 'he answer is' in pred_str:
            # Boxed only found template, try "answer is" format instead
            pred = [pred_str.split('he answer is')[-1].strip()]
        else:
            # Use boxed answers even if they look like templates
            pred = boxed_answers
    elif ('he answer is' in pred_str):
        pred = [pred_str.split('he answer is')[-1].strip()]
    else:
        program_output = extract_program_output(pred_str)
        if program_output != "":
            # fall back to program
            pred.append(program_output)
        else: # use the last number
            pattern = r'-?\d*\.?\d+'
            ans = re.findall(pattern, pred_str.replace(",", ""))
            if(len(ans) >= 1):
                ans = ans[-1]
            else:
                ans = ''
            if ans:
                pred.append(ans)

    # multiple line
    _pred = []
    for ans in pred:
        ans = ans.strip().split("\n")[0]
        ans = ans.lstrip(":")
        ans = ans.rstrip(".")
        ans = ans.rstrip("/")
        ans = strip_string(ans)
        _pred.append(ans)
    if exhaust:
        return _pred
    else:
        return _pred[-1] if _pred else ""

def extract_math_answer(question, reasoning):
    """Extracts all valid math answers from reasoning text."""
    answer = []
    for ans in extract_answer(reasoning, exhaust=True):
        # Handle NaN values in question
        if isinstance(question, str) and 'separated by commas' in question and all(ch not in ans for ch in '()[]'):
            answer.extend([a.strip() for a in ans.split(",")])
        elif regex.search(r"\\text\{\s*and\s*\}", ans):
            answer.extend([a.strip() for a in regex.sub(r"\\text\{\s*and\s*\}", "[SEP]", ans).split("[SEP]")])
        else:
            answer.append(ans.strip())
    return answer

def extract_last_single_answer(question, reasoning):
    """Wrapper to extract the single best answer from reasoning."""
    return extract_answer(reasoning, exhaust=False)

