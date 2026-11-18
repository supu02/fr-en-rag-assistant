# Writes evaluation/questions.jsonl. Run: python evaluation/make_questions.py evaluation/questions.jsonl
# Questions and reference answers were drafted with an AI assistant from the
# source pages (facts checked against data/corpus/chunks.jsonl); review them before trusting them.
# Each topic has a French and an English question with the same content.
import json
import sys

T = []  # (topic, type, gold_pages, fr_q, fr_ref, en_q, en_ref)
def add(*a): T.append(a)

# ---------- Renting (fact) ----------
add("t01", "fact", ["F1168", "F32360"],
    "Je loue un appartement vide dans une commune en zone tendue. Combien de temps de préavis dois-je donner à mon propriétaire pour partir ?",
    "1 mois, à condition d'indiquer dans la lettre de congé que le logement est en zone tendue (sinon c'est 3 mois).",
    "I rent an unfurnished flat in a town classed as a 'zone tendue'. How much notice must I give my landlord before I move out?",
    "1 month, as long as the notice letter says the flat is in a zone tendue (otherwise it is 3 months).")
add("t02", "fact", ["F1168", "F32360", "F1165"],
    "J'habite un studio meublé et je veux déménager. Quel préavis dois-je respecter ?",
    "1 mois, à tout moment, même pour un bail de 9 mois.",
    "I live in a furnished studio and want to move out. What notice period applies?",
    "1 month, at any time, even with a 9-month lease.")
add("t03", "fact", ["F31269", "F1165"],
    "Pour un logement loué vide, quel est le montant maximum du dépôt de garantie ?",
    "1 mois de loyer hors charges (si le loyer est payable d'avance chaque mois).",
    "For an unfurnished rental, what is the maximum security deposit the landlord can ask for?",
    "One month's rent excluding charges (when rent is paid monthly in advance).")
add("t04", "fact", ["F31269", "F33671"],
    "J'ai rendu les clés et l'état des lieux de sortie est identique à celui d'entrée. Dans quel délai le propriétaire doit-il me rendre le dépôt de garantie ?",
    "Dans un délai maximum de 1 mois à partir de la remise des clés (2 mois si l'état des lieux de sortie n'est pas conforme).",
    "I handed back the keys and the move-out inspection matches the move-in one. How long does the landlord have to return my deposit?",
    "At most 1 month from when the keys are returned (2 months if the move-out inspection does not match).")
add("t05", "fact", ["F31269"],
    "Mon propriétaire rend le dépôt de garantie en retard. Ai-je droit à une compensation ?",
    "Oui : le montant à rendre est augmenté de 10 % du loyer mensuel hors charges pour chaque mois de retard commencé (après mise en demeure).",
    "My landlord is late returning my deposit. Am I entitled to any compensation?",
    "Yes: the amount owed increases by 10% of the monthly rent (excluding charges) for each month of delay started (after a formal notice).")
add("t06", "fact", ["F31629", "F31269"],
    "Est-ce que je peux ne pas payer mon dernier mois de loyer et dire au propriétaire de le garder sur le dépôt de garantie ?",
    "Non, c'est interdit : le locataire doit payer tout le loyer et les charges jusqu'à la fin du bail.",
    "Can I skip my last month's rent and tell the landlord to keep it from the deposit instead?",
    "No, that is illegal: the tenant must pay all rent and charges until the end of the lease.")
add("t07", "fact", ["F929"],
    "Mon propriétaire veut récupérer l'appartement vide que je loue. Combien de temps à l'avance doit-il me prévenir ?",
    "Le locataire doit recevoir la lettre de congé au moins 6 mois avant la fin du bail (3 mois pour un logement meublé).",
    "My landlord wants to take back the unfurnished flat I rent. How far in advance must he tell me?",
    "The tenant must receive the notice at least 6 months before the end of the lease (3 months for a furnished flat).")
add("t08", "fact", ["F34759", "F1165"],
    "Combien de temps peut durer un bail mobilité, et le propriétaire peut-il demander un dépôt de garantie ?",
    "De 1 à 10 mois, non renouvelable ; le dépôt de garantie est interdit.",
    "How long can a 'bail mobilité' (short-term furnished lease) last, and can the landlord ask for a deposit?",
    "From 1 to 10 months, not renewable; a deposit is not allowed.")
add("t09", "fact", ["F2449"],
    "Je veux sous-louer mon appartement pendant l'été. Ai-je le droit ?",
    "Seulement avec l'accord écrit du propriétaire, qui porte aussi sur le loyer ; le loyer au m² de la sous-location ne peut pas dépasser celui du locataire.",
    "I want to sublet my flat for the summer. Am I allowed to?",
    "Only with the landlord's written agreement, which also covers the rent; the sublet rent per m² cannot be higher than the tenant's rent.")
add("t10", "fact", ["F34736", "F31272"],
    "Pendant quelle période de l'année un locataire ne peut-il pas être expulsé ?",
    "Pendant la trêve hivernale, du 1er novembre au 31 mars de l'année suivante.",
    "During which period of the year can a tenant not be evicted?",
    "During the winter truce, from 1 November to 31 March of the following year.")
add("t11", "fact", ["F2693"],
    "Mon bail dit que les animaux sont interdits. Est-ce que je peux quand même avoir un chat ?",
    "Oui : le locataire peut avoir des animaux de compagnie s'il respecte la tranquillité du voisinage ; le bail peut seulement interdire les chiens dangereux de 1re catégorie.",
    "My lease says pets are not allowed. Can I still have a cat?",
    "Yes: a tenant can keep pets if neighbours are not disturbed; the lease can only ban category-1 dangerous dogs.")
add("t12", "fact", ["F2889", "F1686"],
    "Mon propriétaire veut me faire payer des frais parce que j'ai payé le loyer avec 5 jours de retard. Est-ce légal ?",
    "Non, le propriétaire ne peut facturer aucun frais ni pénalité de retard de paiement du loyer.",
    "My landlord wants to charge me a fee because I paid the rent 5 days late. Is that legal?",
    "No, the landlord cannot charge any late-payment fee or penalty on rent.")
add("t13", "fact", ["F479"],
    "Pendant combien de temps un propriétaire peut-il réclamer des loyers impayés, même après mon départ ?",
    "Pendant 3 ans.",
    "For how long can a landlord claim unpaid rent, even after I have moved out?",
    "For 3 years.")
add("t14", "fact", ["F1703"],
    "J'ai signé un bail hier mais j'ai trouvé un autre logement. Y a-t-il un délai pour me rétracter ?",
    "Non, il n'y a aucun délai de rétractation après la signature d'un bail d'habitation ; il faut donner congé selon les règles.",
    "I signed a lease yesterday but found another flat. Is there a cooling-off period to cancel?",
    "No, there is no cooling-off period after signing a residential lease; you have to give notice under the normal rules.")

# ---------- Housing aid (fact) ----------
add("t15", "fact", ["F1563"],
    "Je suis étudiant et je loue une chambre. Est-ce que je peux toucher une aide au logement ?",
    "Oui, sous conditions, un étudiant peut bénéficier d'une aide au logement (APL, ALF ou ALS).",
    "I'm a student renting a room. Can I get housing benefit to help pay the rent?",
    "Yes, under conditions a student can receive a housing allowance (APL, ALF or ALS).")
add("t16", "fact", ["F18490"],
    "Quel est le montant maximum de l'avance Loca-Pass pour payer le dépôt de garantie ?",
    "1 200 € maximum.",
    "What is the maximum amount of the Loca-Pass advance for paying a security deposit?",
    "€1,200 at most.")
add("t17", "fact", ["F32384"],
    "Je loue l'appartement de mes parents. Puis-je recevoir l'APL ?",
    "Non, pas d'APL, ALF ou ALS si le propriétaire est un ascendant (ou descendant) du locataire ou de son conjoint.",
    "I rent my parents' flat. Can I receive APL housing benefit?",
    "No, there is no APL, ALF or ALS when the landlord is a parent (or child) of the tenant or of the tenant's partner.")

# ---------- Leave (fact) ----------
add("t18", "fact", ["F2258"],
    "Combien de jours de congés payés est-ce qu'on gagne par mois de travail ?",
    "2,5 jours ouvrables par mois, soit 30 jours ouvrables (5 semaines) par an.",
    "How many days of paid holiday do employees earn per month worked?",
    "2.5 working days per month, i.e. 30 working days (5 weeks) per year.")
add("t19", "fact", ["F12410"],
    "Il me reste des jours de congés payés non pris. Puis-je les reporter à l'année suivante ?",
    "En principe non, il n'y a pas de droit au report ; c'est possible seulement en cas d'accord entre l'employeur et le salarié (ou cas particuliers).",
    "I have unused paid holiday days. Can I carry them over to next year?",
    "In principle no, there is no right to carry them over; it is possible only if employer and employee agree (or in special cases).")
add("t20", "fact", ["F34154"],
    "Je me marie le mois prochain. Combien de jours de congé mon employeur doit-il me donner ?",
    "4 jours (même durée pour un Pacs), ou plus si la convention collective le prévoit.",
    "I'm getting married next month. How many days off must my employer give me?",
    "4 days (same for a PACS), or more if the collective agreement says so.")
add("t21", "fact", ["F2278"],
    "Mon père est décédé. Combien de jours de congé ai-je en tant que salarié ?",
    "3 jours, ou plus si la convention collective ou un accord le prévoit.",
    "My father has died. How many days of leave do I get as an employee?",
    "3 days, or more if the collective agreement provides it.")
add("t22", "fact", ["F3156", "F12647"],
    "Quelle est la durée du congé de paternité pour la naissance d'un enfant ?",
    "25 jours calendaires (après le congé de naissance de 3 jours), dont 4 jours obligatoires.",
    "How long is paternity leave for the birth of one child?",
    "25 calendar days (after the 3-day birth leave), of which 4 days are compulsory.")
add("t23", "fact", ["F2265"],
    "J'attends mon premier enfant. Combien de temps dure le congé maternité ?",
    "16 semaines : 6 semaines avant l'accouchement et 10 semaines après.",
    "I'm expecting my first child. How long is maternity leave?",
    "16 weeks: 6 weeks before the birth and 10 weeks after.")
add("t24", "fact", ["F33902"],
    "Je déménage la semaine prochaine. Ai-je droit à un jour de congé pour déménagement ?",
    "Il n'existe pas de congé légal pour déménagement ; seulement si la convention collective, un accord ou un usage le prévoit.",
    "I'm moving house next week. Am I entitled to a day off for moving?",
    "There is no legal right to moving leave; only if the collective agreement, a company agreement or custom provides it.")
add("t25", "fact", ["F151"],
    "Mon enfant de 6 ans est malade. Combien de jours d'absence puis-je prendre par an ?",
    "3 jours par an (5 jours si l'enfant a moins d'1 an ou si le salarié a au moins 3 enfants de moins de 16 ans).",
    "My 6-year-old child is sick. How many days off can I take per year?",
    "3 days per year (5 days if the child is under 1 or the employee has at least 3 children under 16).")
add("t26", "fact", ["F2381"],
    "Je voudrais prendre un congé sabbatique. Combien de temps peut-il durer et faut-il de l'ancienneté ?",
    "De 6 à 11 mois ; il faut au moins 36 mois d'ancienneté dans l'entreprise (entre autres conditions).",
    "I'd like to take a sabbatical. How long can it last, and do I need a minimum length of service?",
    "From 6 to 11 months; you need at least 36 months' service in the company (among other conditions).")
add("t27", "fact", ["F37482"],
    "Est-ce que j'accumule des congés payés pendant un arrêt maladie (maladie non professionnelle) ?",
    "Oui, dans la limite de 2 jours ouvrables par mois, soit 24 jours ouvrables par an.",
    "Do I still earn paid holiday while on sick leave for a non-work-related illness?",
    "Yes, up to 2 working days per month, i.e. 24 working days per year.")

# ---------- Leaving a job (fact) ----------
add("t28", "fact", ["F2883"],
    "Si je démissionne de mon CDI, dois-je expliquer pourquoi et puis-je partir tout de suite ?",
    "Pas besoin de donner un motif ; mais en règle générale il faut effectuer un préavis (sauf dispense).",
    "If I resign from my permanent job, do I have to give a reason, and can I leave immediately?",
    "No reason is needed; but as a rule you must work a notice period (unless excused).")
add("t29", "fact", ["F19030"],
    "Après avoir signé une rupture conventionnelle, puis-je encore changer d'avis ?",
    "Oui, l'employeur et le salarié ont un droit de rétractation de 15 jours calendaires.",
    "After signing a 'rupture conventionnelle' (mutual termination), can I still change my mind?",
    "Yes, both employer and employee have a 15-calendar-day right to withdraw.")
add("t30", "fact", ["F89"],
    "Si je démissionne, est-ce que j'aurai droit au chômage ?",
    "En principe non, la démission ne donne pas droit à l'ARE, sauf cas particuliers (démission considérée comme légitime, reliquat de droits...).",
    "If I resign, will I get unemployment benefits?",
    "In principle no, resigning gives no right to unemployment benefit (ARE), except in special cases (a resignation accepted as legitimate, remaining rights...).")
add("t31", "fact", ["F86", "F21789"],
    "J'ai signé mon reçu pour solde de tout compte. Puis-je encore contester les sommes ?",
    "Oui, dans un délai de 6 mois après la signature, par lettre recommandée à l'employeur.",
    "I signed my final pay statement ('solde de tout compte'). Can I still dispute the amounts?",
    "Yes, within 6 months after signing, by registered letter to the employer.")
add("t32", "fact", ["F987", "F371"],
    "Comment est calculée l'indemnité légale de licenciement ?",
    "Il faut au moins 8 mois d'ancienneté ; minimum 1/4 de mois de salaire par année d'ancienneté jusqu'à 10 ans, puis 1/3 de mois par année au-delà.",
    "How is the legal severance pay for dismissal calculated?",
    "You need at least 8 months' service; at least 1/4 of a month's salary per year of service up to 10 years, then 1/3 of a month per year after that.")
add("t33", "fact", ["F2855"],
    "Je suis licencié après 3 ans dans l'entreprise. Combien dure mon préavis ?",
    "2 mois (ancienneté d'au moins 2 ans), sauf disposition plus favorable.",
    "I'm being dismissed after 3 years with the company. How long is my notice period?",
    "2 months (2 years' service or more), unless a more favourable rule applies.")
add("t34", "fact", ["F31209"],
    "Que se passe-t-il si un salarié ne vient plus travailler sans prévenir ?",
    "C'est un abandon de poste, qui peut être considéré comme une démission ; l'employeur doit le mettre en demeure de reprendre son poste dans un délai d'au moins 15 jours calendaires.",
    "What happens if an employee simply stops coming to work without telling anyone?",
    "It is job abandonment and can be treated as a resignation; the employer must send a formal notice giving at least 15 calendar days to come back.")

# ---------- Contracts (fact) ----------
add("t35", "fact", ["F38", "F36"],
    "Combien de fois un CDD peut-il être renouvelé et quelle est sa durée maximale ?",
    "2 renouvellements maximum (sauf accord de branche) ; durée maximale de 18 mois dans le cas général, renouvellements compris.",
    "How many times can a fixed-term contract (CDD) be renewed, and what is its maximum length?",
    "At most 2 renewals (unless a sector agreement says otherwise); 18 months maximum in the general case, renewals included.")
add("t36", "fact", ["F40", "F41"],
    "À la fin de mon CDD, ai-je droit à une prime ? De combien ?",
    "Oui, l'indemnité de fin de contrat (prime de précarité) est au minimum de 10 % de la rémunération brute totale.",
    "At the end of my fixed-term contract, do I get a bonus? How much?",
    "Yes, the end-of-contract allowance ('prime de précarité') is at least 10% of the total gross pay.")
add("t37", "fact", ["F1945"],
    "Puis-je avoir deux emplois salariés en même temps ?",
    "Oui, mais sans dépasser 10 heures par jour et 48 heures par semaine (ou 44 heures en moyenne sur 12 semaines).",
    "Can I have two salaried jobs at the same time?",
    "Yes, but without working more than 10 hours a day and 48 hours a week (or 44 hours on average over 12 weeks).")

# ---------- Date arithmetic (needs the tool) ----------
add("t38", "date", ["F31269"],
    "Logement vide, état des lieux de sortie conforme. J'ai rendu les clés le 15 janvier 2027. Jusqu'à quelle date le propriétaire a-t-il pour me rendre mon dépôt de garantie ?",
    "1 mois après la remise des clés : au plus tard le 15 février 2027.",
    "Unfurnished flat, the move-out inspection matches the move-in one. I returned the keys on 15 January 2027. By what date must the landlord give back my deposit?",
    "1 month after the keys are returned: by 15 February 2027.")
add("t39", "date", ["F1168", "F32360"],
    "Je loue un meublé. Mon propriétaire a reçu ma lettre de congé le 30 janvier 2027. Quel est le dernier jour de mon préavis ?",
    "Préavis de 1 mois : il se termine le 28 février 2027 (pas de 30 février, donc dernier jour du mois).",
    "I rent a furnished flat. My landlord received my notice letter on 30 January 2027. What is the last day of my notice period?",
    "1-month notice: it ends on 28 February 2027 (there is no 30 February, so the last day of the month).")
add("t40", "date", ["F19030"],
    "Nous avons signé la convention de rupture conventionnelle le 3 mars 2027. Jusqu'à quelle date puis-je me rétracter ?",
    "15 jours calendaires à partir du lendemain de la signature : jusqu'au 18 mars 2027.",
    "We signed the mutual termination agreement (rupture conventionnelle) on 3 March 2027. Until what date can I withdraw?",
    "15 calendar days starting the day after signing: until 18 March 2027.")
add("t41", "date", ["F86"],
    "J'ai signé mon solde de tout compte le 10 avril 2027. Jusqu'à quelle date puis-je le contester ?",
    "6 mois après la signature : jusqu'au 10 octobre 2027.",
    "I signed my final pay statement (solde de tout compte) on 10 April 2027. Until what date can I dispute it?",
    "6 months after signing: until 10 October 2027.")
add("t42", "date", ["F929"],
    "Mon bail de logement vide se termine le 31 août 2027. Avant quelle date dois-je avoir reçu la lettre si mon propriétaire veut me donner congé ?",
    "Au moins 6 mois avant la fin du bail : au plus tard le 28 février 2027.",
    "My unfurnished lease ends on 31 August 2027. By what date must I receive the letter if my landlord wants to end the lease?",
    "At least 6 months before the end of the lease: by 28 February 2027.")
add("t43", "date", ["F2265"],
    "Mon accouchement est prévu le 10 juin 2027 et c'est mon premier enfant. À quelle date commence normalement mon congé maternité ?",
    "6 semaines avant la date prévue : le 29 avril 2027 (une partie peut être reportée après la naissance sur avis médical).",
    "My baby is due on 10 June 2027 and it's my first child. On what date does my maternity leave normally start?",
    "6 weeks before the due date: 29 April 2027 (part of it can be moved to after the birth with medical approval).")
add("t44", "date", ["F2381"],
    "Je veux partir en congé sabbatique le 1er septembre 2027. Avant quelle date dois-je prévenir mon employeur ?",
    "Au moins 3 mois avant le départ : au plus tard le 1er juin 2027.",
    "I want to start a sabbatical on 1 September 2027. By what date must I tell my employer?",
    "At least 3 months before leaving: by 1 June 2027.")
add("t45", "date", ["F18490"],
    "Je suis entré dans mon logement le 15 octobre 2026. Jusqu'à quand puis-je demander l'avance Loca-Pass ?",
    "Au plus tard 2 mois après l'entrée dans le logement : jusqu'au 15 décembre 2026.",
    "I moved into my flat on 15 October 2026. Until when can I apply for the Loca-Pass advance?",
    "No later than 2 months after moving in: until 15 December 2026.")

# ---------- Out of scope: the right answer is "not in my sources" ----------
OOS_FR = "Les sources ne couvrent pas cette question : l'assistant doit le dire et ne pas inventer de réponse."
OOS_EN = "The sources do not cover this question: the assistant should say so and not make up an answer."
add("t46", "unanswerable", [], "Comment renouveler mon passeport ?", OOS_FR,
    "How do I renew my passport?", OOS_EN)
add("t47", "unanswerable", [], "Quel est le taux de TVA appliqué dans les restaurants ?", OOS_FR,
    "What VAT rate applies to restaurant meals?", OOS_EN)
add("t48", "unanswerable", [], "Comment faire la carte grise d'une voiture d'occasion que je viens d'acheter ?", OOS_FR,
    "How do I register a second-hand car I just bought?", OOS_EN)
add("t49", "unanswerable", [], "À partir de quel âge peut-on voter en France ?", OOS_FR,
    "From what age can you vote in France?", OOS_EN)
add("t50", "unanswerable", [], "Combien coûte une consultation chez un médecin généraliste et combien est remboursé ?", OOS_FR,
    "How much does a GP appointment cost and how much is reimbursed?", OOS_EN)

assert len(T) == 50 and len({t[0] for t in T}) == 50
out = sys.argv[1]
with open(out, "w", encoding="utf-8") as fh:
    for topic, qtype, gold, fq, fr, eq, er in T:
        fh.writelines(json.dumps({
                "id": f"{topic}-{lang}", "topic": topic, "lang": lang, "type": qtype,
                "question": q, "reference": ref, "gold_pages": gold,
                "answerable": qtype != "unanswerable", "needs_tool": qtype == "date",
            }, ensure_ascii=False) + "\n" for lang, q, ref in (("fr", fq, fr), ("en", eq, er)))
print("wrote", 2 * len(T), "questions to", out)
