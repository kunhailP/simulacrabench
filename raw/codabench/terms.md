# Competition Rules

By clicking "I agree" and/or registering for the Competition, the Participant
represents that they have read, understood, and agree to be legally bound by the
terms of this Participant Agreement, which constitutes a binding contract
between the Participant and the Competition organizers. The Participant further
agrees that these terms are enforceable against them individually and, where
applicable, against any institution on whose behalf they submit.

## Prohibition on re-identification and reverse engineering

The microdata underlying the Competition are provided to the Competition
organizers by UNICEF, the Office of the United Nations High Commissioner for
Refugees (UNHCR), and the World Bank (collectively, the "Data Providers") under
separate confidentiality agreements. The microdata are not released to
participants.

Any attempt to exfiltrate, reconstruct, re-identify, or otherwise derive the
underlying records or any personal or sensitive information from them — whether
directly or indirectly, and including via side channels such as loss-curve
encoding, deliberately inflated error messages, out-of-spec container writes,
linkage with external datasets, or any other technical or analytical means — is
strictly prohibited.

By registering for and participating in the Competition, each Participant agrees
that this prohibition is a material term of a binding agreement between the
Participant and the Competition organizers. Each of the Data Providers is an
express intended third-party beneficiary of this provision and is entitled to
enforce it directly against any Participant, in its own name. Breach constitutes
not only grounds for immediate disqualification and forfeiture of any prizes,
standings, publications, or recognitions following an organizer vote, but also a
breach of contract entitling the Competition organizers and any Data Provider to
seek injunctive relief, damages, and any other remedies available at law or in
equity. Nothing in this provision, and no enforcement action by any Data
Provider hereunder, shall constitute or be considered a limitation upon or a
waiver of any privileges and immunities of any Data Provider, all of which are
expressly reserved.

## Participation and team composition

1. Participation is open to individuals and teams from academia, industry, and
   independent research, except where precluded by sanctions or applicable law.
2. An individual may appear on at most one team. Multi-team collusion, including
   coordinated submission splitting, is grounds for disqualification of all
   involved teams and is decided by organizer vote.
3. Team membership can change up to submission of the final test model, at which
   point it is frozen.

## Submissions

4. A submission is a ZIP archive with `main.py` at its root, defining
   `predict(frame, schema)`, optionally accompanied by `requirements.txt`,
   `models.txt`, and any files the submission loads. Reproducibility is the participant's responsibility — a
   submission that fails to load, that returns output violating the contract, or
   that exceeds the phase's wall-clock budget on the single H100 the harness
   runs, receives no score.
5. Outbound network access from the submission container is disabled. Pretrained
   weights must either be packaged in the submission or declared through the
   supported `models.txt` mechanism for organizer-side prefetch, within the
   archive-size, model-download, and compute limits.
6. Development has a quota of one leaderboard submission per team per day,
   each scored on every respondent assigned the `DEV` role. Test has a hard cap
   of one submission per team, scored on every respondent assigned the `TEST`
   role. Role assignments are fixed when the dataset is built, so submissions
   within a phase receive and are scored on the same applicable rows; neither
   role is sampled or thinned.
7. Each Test submission must be accompanied by a 4-page method description. At
   the award stage, the top 3 must additionally supply source code under a
   license permitting non-commercial research reproduction.

## Data handling and integrity

8. The microdata are not released to participants. Any attempt to exfiltrate
   records through submitted code — including via side channels such as
   loss-curve encoding, deliberately inflated error messages, or out-of-spec
   container writes — is forbidden. It is grounds for disqualification following
   an organizer vote.
9. Use of external pretrained models is permitted provided weights are publicly
   downloadable at a fixed commit hash specified by the participant before the
   Test phase opens, and licensed with a license permitting non-commercial
   research reproduction. Private fine-tunes on proprietary datasets unrelated
   to the competition data are permitted.
10. Submitted code may be reviewed by the organizers for integrity and rule
    compliance.

## Edge cases explicitly addressed

11. Ties on the primary metric within paired-bootstrap significance (at a
    pre-registered threshold) lead to shared prize money. Test scores are
    returned exactly, so the threshold accounts for sampling variability alone.
12. If the harness itself fails on a specific submission due to organizer-side
    infrastructure issues (documented in the Codabench logs), the team receives
    a no-charge retry.
13. Entry is permitted through the end of the Development phase; no team is
    penalized for beginning late, provided they meet the Test deadline.
14. Ambiguities not covered above are resolved by majority vote of the
    organizing committee; decisions are broadcast to all teams.

## Authorship and communication

15. Winning teams will be invited to contribute method descriptions to the
    proceedings paper, with authorship following contribution and approval of
    the final manuscript. The top 3 teams are invited for named authorship; all
    compliant participants are listed in the acknowledgments.
16. All communication runs through the dedicated competition email
    <neurips-un-benchmark@stanford.edu>. Rule and deadline updates are broadcast
    to all registered teams and on the competition website.