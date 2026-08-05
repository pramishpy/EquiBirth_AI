# EquiBirth AI: speaker script

Everything I say out loud, in order, for the 13-slide deck.

The lines inside the `say` blocks are the actual words. They are written the way people talk, not
the way papers read, so they should feel natural in your mouth. Everything outside those blocks is
a note to you and is never spoken.

**About 1,630 spoken words. That is 9:41 at your usual pace of 170 words per minute, and 9:10 if
you speak fast. If you speak slowly, near 160 words per minute, it creeps to about 10:20, so on a
slow day take one item off the cut list at the bottom.** The two-second pauses on slides 2 and 6
are already counted.

`make presentation-check` re-counts this every time and fails if an edit pushes it past 9:40 at
your usual pace.

---

## How to use this before the talk

1. **Read it out loud three times sitting down.** Not silently. Silent reading teaches you nothing
   about where you run out of breath.
2. **Then read it out loud three times standing up, with the slides open.** Standing changes your
   breathing. Practice the version you will actually perform.
3. **Memorize only the bolded lines.** Those are the ten or so sentences that carry the talk. For
   everything else, knowing the idea in order is enough. If you memorize every word you will sound
   like you are reciting, and one forgotten word will derail you.
4. **Time yourself twice.** Write your two times at the bottom of this file. If you are over 10:00,
   use the cut list at the end. Do not speed up. Speeding up is what makes people impossible to
   follow.
5. **Practice the two pauses.** They are marked `[PAUSE]`, on slides 2 and 6. Count two full
   seconds in your head. Two seconds feels like forever to you and like confidence to the room.
   The timing already includes them, so you are not stealing from anything by taking them.

One thing that helps introverts more than anything else: you are not performing, you are
explaining. Every single person in that room knows less about this project than you do. You are
the only one who spent nine weeks on it. That is not arrogance, it is just true, and it means you
cannot be caught out.

### The whole talk in one sentence

If you forget everything else, this is the spine:

> A suspiciously good number, why it was fake, what the honest number is, what it revealed about
> inequity, and where we turned out to be wrong.

Every slide is one step along that line. If you lose your place mid-talk, ask yourself which of
those five things you just finished, and go to the next one.

### Where you should be at each checkpoint

Four glances at the clock, not thirteen. If you are on the right slide at these four moments you
will finish on time without ever rushing.

| When the clock says | You should be starting |
|---|---|
| 1:40 | Slide 4, the timeline |
| 3:25 | Slide 6, the weaker model |
| 6:00 | Slide 9, education |
| 7:50 | Slide 11, the decomposition |

More than 30 seconds behind at any of these? Take the matching cut from the list at the bottom.
Do not speed up.

---

# 1. Title: EquiBirth AI

**Time: 0:00 to 0:20. On screen:** project name, the question, and the line "Our best result was
not our highest score."

Before you speak: put both feet flat, look at the back wall, breathe out once. Then start.

```say
Hi everyone. I'm Parshv, and this is EquiBirth AI.

We started with a simple question. Can you predict preterm birth from a U.S. birth
certificate? And if you can, is the model fair?

Nine weeks later, the most interesting thing we found was a number we decided not to
report. Let me show you what I mean.
```

**Delivery:** do not read the slide out loud. The room can read. Your job is the last line, which
is a promise that something is coming.

---

# 2. The hook: you should not be impressed

**Time: 0:20 to 0:54. On screen:** "If I showed you a preterm-birth model scoring 0.92, you should
not be impressed."

```say
Here is the number we did not report.

If I told you we had a preterm-birth model scoring zero point nine two, most people in
this room would be impressed. Zero point nine two is a very good score.

Please don't be impressed.

[PAUSE]

Instead, ask me one question. Ask what the model was allowed to look at.

Because a model can look brilliant for one very boring reason. Somebody accidentally
handed it the answer.

That is the trap. Most of our nine weeks went into not falling into it.
```

**Delivery:** "Please don't be impressed" is the moment the room decides whether to pay attention.
Say it slightly quieter than the sentence before, not louder. Then the pause. Count two.

**Do not** explain leakage yet. Slide 4 does that. Resist it.

---

# 3. Why this matters: common, and unevenly distributed

**Time: 0:54 to 1:41. On screen:** 8.57%, 12.28% vs 7.54%, 4.74 points, plus the race-framing
panel.

```say
So first, why this is worth the trouble.

Preterm birth means the baby arrives before thirty seven weeks. It happens in about
eight and a half percent of U.S. single births, and it is the leading cause of death
in newborns.

Now the middle number. For non-Hispanic Black mothers that rate is twelve point three
percent. For White mothers, seven point five. A gap of four point seven points.

And nobody estimated that from a survey. That is every single birth certificate filed
in the country over seven years. Almost twenty five million of them.

One thing I want to be clear about, and I mean it for the whole talk. When race shows up
in our work, it stands in for exposure to structural conditions and to racism. It is
never a biological cause.
```

**Delivery:** point at the middle number when you get to it. On the last paragraph, stop moving,
face the room, and slow down. It should sound like a commitment, not a disclaimer.

---

# 4. The trap: the certificate knows how it ended

**Time: 1:41 to 2:37. On screen:** the pregnancy timeline, split into what is known before delivery
and what is only recorded at or after it.

```say
So, back to that trap. It has a name. Data leakage.

A birth certificate gets filled in after the baby is born. So it holds two very
different kinds of information, and this timeline splits them at the delivery.

Everything on the left is known during the pregnancy. Her age, her BMI, a prior preterm
birth. Everything on the right only gets written down once the baby is out. Birthweight.
Apgar score. Intensive care.

And here is why that matters. A baby born at thirty weeks weighs about one and a half
kilos. So birthweight is not a risk factor for preterm birth. It is a consequence of it.

My favorite one is that steroid injection on the right. Doctors give it when they
already expect an early delivery. So it is not a prediction. It is a record of one.

Give a model the right-hand side and it is not forecasting. It is reading the ending.
```

**Delivery:** use your hands for this one. Left hand for the left side of the timeline, right hand
for the right. It makes the split obvious without a word of extra explanation.

**"It is reading the ending"** is a bolded line. Land it, then move on.

---

# 5. The trap, quantified: 0.939, 0.917, 0.659

**Time: 2:37 to 3:25. On screen:** three horizontal bars.

```say
We did not want to just assert that, so we measured it. Three models, scored on the same
held-out births.

The top bar is us cheating on purpose. We handed our own model the right-hand side of
that timeline. Zero point nine three nine.

The middle bar is the one I find genuinely unsettling. That model got only the banned
information. Zero point nine one seven. It has no idea this woman was ever pregnant. It
is reading the delivery record.

And the bottom bar is us playing fair. Before-delivery only. Zero point six five nine.

That drop is the most important result in this project.

And we did not check it once and then trust ourselves. Our build now fails automatically
if any model trained on the banned information scores above zero point five five.
```

**Delivery:** walk the bars top to bottom with your hand. When you say "it is reading the delivery
record," pause for half a beat. That is the sentence that makes leakage click for people who have
never heard the word.

---

# 6. Our hardest decision: we chose the weaker model

**Time: 3:25 to 4:26. On screen:** the 0.684 to 0.654 tradeoff, plus the reverse-causality
explanation.

This is the most important slide in the talk. Slow down. Take an extra breath before you start.

```say
But automatic checks only catch the obvious cheating. This next one they missed.

Three conditions. Eclampsia, gestational hypertension, gestational diabetes. All
diagnosed during the pregnancy, before delivery. So every test I just described says
keep them. They are completely legal.

But think about what happens. A mother has severe eclampsia at thirty three weeks. The
team delivers early, because that is the right thing to do. So the condition is not
predicting the early birth. It is causing it. The arrow points the wrong way. The model
would learn "a doctor decided to deliver," and we would call that a prediction.

So we took them out, and it cost us. On our development sample, we dropped from zero
point six eight four to zero point six five four.

And here is the part I want to own. Nobody outside our team would ever have known. We
could have kept the better number and passed every check.

We chose the weaker model, because it was the more honest model.

[PAUSE]
```

**Delivery:** the closing line is the single line most worth memorizing word for word. Say it
looking at the room, not at the screen. Then stop. Let it sit for a second before you click on.

---

# 7. Our result: the honest answer is 0.656

**Time: 4:26 to 5:11. On screen:** five models compared, with the compute caveat called out.

```say
So this is where we landed. Zero point six five six, from a tuned tree model. The bars
show a stricter score for the same models, and we win on both.

And we did try the fancy options. The bottom two orange bars are deep learning.

The trees beat them, but I want to be careful, because this is easy to overclaim. We
trained those neural models on thirty thousand rows on a laptop. The trees got half a
million. Not a fair fight, and we say so on the chart.

So the honest claim is narrow. Under our compute budget the simpler model won, and we
picked it because we could audit it. Not "deep learning does not work here." We never
gave it a fair enough shot.
```

**Delivery:** point at the two **bottom** orange bars, not to the right. They are the deep learning
models. Volunteering the weakness of your own comparison is what makes the rest of your numbers
believable, so do not skip that paragraph.

**Why the two numbers:** the title says 0.656 and the bars say 0.1720. They are two different
measures of the same model. 0.656 is AUC-ROC, which is what people quote; 0.1720 is AUC-PR, which
is the honest one when the outcome is rare. The script says this in plain words so nobody in the
audience thinks the numbers disagree. **If asked:** "0.656 is the standard score, 0.172 is the
precision-recall score, same model, and the precision one is the harder test."

---

# 8. Why 0.656 is the finding: the ceiling is in the data

**Time: 5:11 to 5:58. On screen:** per-field signal on the left, the training-size plateau on the
right.

```say
That is a modest number, so let me answer the obvious question. Did we just not try hard
enough? These charts say no.

On the left, we tested every field on its own. A coin flip scores zero point five. The
strongest single field on the whole form gets you to zero point five seven. Everything
else is weaker.

On the right, brute force. Five times the training data. The line is basically flat.

So more rows and bigger models cannot manufacture signal the document does not contain.
The ceiling is not in our code. It is in the data.

Which has a real consequence. Do not expect a form like this to rank individual risk.
Ours does not, and that is exactly why we say this is not a clinical tool.
```

**Delivery:** "The ceiling is not in our code, it is in the data" is a bolded line. It reframes a
weak-sounding score as a measurement, and it is the intellectual center of the project.

---

# 9. Equity: education does not close the gap

**Time: 5:58 to 6:51. On screen:** the outcome disparity on the left, the education strata on the
right.

```say
So if the certificate cannot tell you much about an individual, what is it good for?
Population patterns. And this is where it got interesting.

On the left is that gap from the start of the talk. The obvious objection, and it is a
good one. Maybe this is really about money and education, and race is standing in for
class.

So we checked. The right side splits both groups by education, from no high school up to
a doctorate. If class were the whole story, those lines would come together as you move
right.

They do not. The gap holds at every level, between three and four and a half points. The
lines just run parallel.

And here is the comparison that stopped me. Black mothers with a doctorate have a higher
preterm rate than White mothers who stopped at high school.

Education does not close this gap.
```

**Delivery:** the doctorate comparison is the line the room will remember and possibly repeat.
Say it slowly, and then say the four-word sentence after it slowly too.

**If someone pushes back afterward:** be honest that this is associational. It does not prove a
mechanism. What it does is rule out the simplest alternative explanation.

---

# 10. The result we did not expect: our fairness hypothesis was wrong

**Time: 6:51 to 7:51. On screen:** more detection on the left, 2.74x more flagging on the right.

```say
So that is the gap in the world. Our model did not create it. But we had to ask whether
it makes it worse. And we got the opposite of what we expected.

Before we ran anything, we wrote our prediction down. We thought the model would
under-detect for Black mothers and miss more of their preterm births. The usual worry.

The left chart says we were wrong. It catches more of their real preterm births, not
fewer. That is the flattering half.

The right chart is the cost. Black mothers get flagged two point seven four times as
often. More follow-ups, more anxiety, on one group. To be fair, most of that is the real
rate being higher.

But both are true at once. Catching more of a group's cases while flagging them nearly
three times as often is not the same as treating them equitably.

I will not resolve that today. Fairness is not one number, and pretending it is is the
shortcut we refused a few slides ago.
```

**Delivery:** the temptation is to present the left chart as good news. Do not. Give both charts
the same tone of voice. The whole point of the slide is that they arrive together.

---

# 11. What explains it: measured factors explain under half

**Time: 7:51 to 8:33. On screen:** the decomposition, 19% to 43%.

```say
One last result. Back to that gap in preterm rates. How much of it can the things on this
form account for?

We took education, insurance, WIC, and medical history, and asked how much of the gap
runs through them. Somewhere between nineteen and forty three percent. Most of it is not
in there at all.

And there is a catch that makes even forty three percent too flattering. Education and
insurance are not separate explanations. They are shaped by the same structural
conditions. So they are not rivals to the racism explanation. They are part of it.

The plain reading. Go after only the economic factors on this form, and you close under
half the gap at best.
```

**Delivery:** "they are not rivals to the racism explanation, they are part of how it works" is the
sentence that matters. The technical word is mediation, but you do not need it out loud.

**Watch out:** this chart's axis says the gap is **4.91** points, while slides 3 and 9 said 4.74.
Both are correct: 4.74 is every birth in the country, 4.91 is the smaller modelling sample this
particular analysis ran on. That is why the script says "that gap in the outcome" rather than
repeating a number here. **If someone asks:** "4.74 is the full population, 4.91 is the modelling
subsample this decomposition used. Same gap, slightly different denominator."

---

# 12. Reflection: what nine weeks actually taught us

**Time: 8:33 to 9:17. On screen:** four numbered lessons.

Speak this one from memory if you can. Look at the room, not the slide.

```say
So, four things I am taking away.

One. A great score is a suspect. Zero point nine one seven was there all summer.

Two. A number can be true and still useless. Ours could not tell any mother anything she
did not already know.

Three. Fairness is not one number. Ours looked generous and unfair.

And four, the one I will actually use again. Write down what you expect before you
measure it. Being publicly wrong about the fairness direction was our most interesting
finding, and we could only be wrong out loud because we wrote it down first.

The line at the bottom of this slide sums it up. The skill we practised most was
deciding what we were not entitled to claim.
```

---

# 13. Closing: we said 0.656, and showed you why

**Time: 9:17 to 9:40. On screen:** the two-line contrast, and "Thank you. Questions?"

```say
So, to close where we started.

A project built to impress you could have stood up and said zero point nine one seven.

We said zero point six five six, and we showed you why.

The hardest engineering in this whole thing was not the model. It was building the guards
that stopped us reporting the better-looking number.

Thank you. I am happy to take questions.
```

**Delivery:** deliver the two-line contrast, then stop completely. Do not add anything. Do not
trail off with "yeah, so, that's it." Stand still and wait. Silence at the end reads as confidence
and it gives the room a beat to start clapping.

---

## If you are running out of time

Cut in this order. Each cut is clean, meaning nothing later in the talk depends on it.

1. **Slide 11 down to one sentence** (saves ~33s): "Measurable factors like education and
   insurance explain under half of the gap, and they are themselves shaped by the same structural
   conditions."
2. **Slide 4's antenatal-steroids example** (saves ~13s): the birthweight example alone already
   makes the point. Drop the steroids paragraph, keep everything else.
3. **Slide 12 down to lessons one and four** (saves ~13s): those two are the ones worth hearing.
4. **Slide 8's opening question** (saves ~8s): start straight at "These two charts are our answer."

Doing all four buys you about a minute. Never cut slides 5, 6, or 10, and do not cut the compute
caveat on slide 7 or the not-a-clinical-tool line on slide 8. Those are honesty points, and they
are the reason the rest of the talk is credible.

## If you finish early

Good. Do not fill the time. Say "I will leave it there so we have room for questions." Finishing
around 9:00 is a strength, not a gap you need to plaster over.

## Questions you will probably get

Answer in two sentences, then stop. Long answers are where prepared speakers come apart.

**"Could this be used in a clinic?"**
No, and we designed it not to be. There is no risk calculator in our app, and the per-person
predictions are too flat to be useful for an individual decision.

**"Isn't 0.656 quite bad?"**
It is modest, and it sits inside the published range for administrative birth data. Our finding is
that the ceiling comes from the document, not from the model.

**"Why is race in the model at all?"**
Because dropping it does not remove the disparity, it just hides our ability to measure it. It
enters as a marker of exposure to structural conditions, and we say that everywhere it appears.

**"Did you try to fix the fairness gap?"**
We priced three mitigations. The most effective one needs different thresholds by race, which is
disparate treatment, so we did not ship it. We reported the tradeoffs instead of picking the one
that looked best.

**"What would you do next?"**
Link birth records to something the certificate cannot see, like neighborhood or environmental
exposure. The ceiling result says extra modeling on this same form will not help.

**"How do you know your model has no leakage?"**
Three automated guards, and the build fails if any of them trips. One trains a detector on the
banned columns and fails above 0.55. There are tests for all three.

## Practice log

| Run | Date | Time | Notes |
|---|---|---|---|
| 1 |  |  |  |
| 2 |  |  |  |
| 3 |  |  |  |
