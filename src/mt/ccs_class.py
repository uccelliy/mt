import numpy as np
import pandas as pd
import torch

def main():

    print(type(symptoms["ADHD"][0]))
    # input is a sentence describe symptom Output is a disease label for the symptom
    example_input = input_map2text(symptoms)
    print(example_input)

symptoms = {
        "Major Depressive Disorder": [
            "I feel sad or empty most of the day",
            "I have lost interest in activities I used to enjoy",
            "I am tired all the time even after resting",
            "I have difficulty concentrating on simple tasks",
            "I feel worthless or guilty without clear reason",
            "I have trouble making everyday decisions",
            "My appetite has changed noticeably",
            "I sleep too much or cannot sleep at all",
            "I think about death frequently",
            "I move and speak more slowly than I used to",
        ],
        "Generalized Anxiety Disorder": [
            "I worry about everything even small things",
            "I cannot control my worrying thoughts",
            "I feel restless and on edge most days",
            "I am tired despite not doing very much",
            "I have difficulty concentrating because of worry",
            "I am irritable with people around me",
            "My muscles feel tense and sore for no reason",
            "I cannot fall asleep because my mind is racing",
            "I feel a constant sense of impending doom",
            "I worry about my health all the time",
        ],
        "ADHD": [
            "I am easily distracted by sounds or movement",
            "I have trouble finishing what I start",
            "I lose important objects regularly",
            "I forget appointments and obligations",
            "I find it hard to sit still during meetings",
            "I interrupt others when they are speaking",
            "I make careless mistakes at work or school",
            "I avoid tasks that require sustained mental effort",
            "I fidget with my hands or feet constantly",
            "I have trouble organizing my daily tasks",
        ],
        "Insomnia Disorder": [
            "I lie awake for hours before falling asleep",
            "I wake up multiple times during the night",
            "I wake up too early and cannot return to sleep",
            "I feel unrefreshed in the morning despite sleeping",
            "I worry about not being able to sleep",
            "I am sleepy during the day from lack of sleep",
            "My sleep is interrupted by physical discomfort",
            "I dread going to bed because of my sleeplessness",
            "I rely on substances to fall asleep",
            "My total sleep time is much less than I need",
        ],
    }
def input_map2text(input_map):
    output = []
    # 讲input_map取出disease以及对应的symptom，每一条sympotm后面加入 disease is {disease},然后加入output list中，
    # 每一条新处理的sympotom+disease之间用换行符分隔开来，最后返回这个output list
    for map_key, map_value_list in input_map.items():
        for map_value in map_value_list:
            output.append("{map_value} disease is <<{map_key}>>\n")
    return output
if __name__ == "__main__":
    main()