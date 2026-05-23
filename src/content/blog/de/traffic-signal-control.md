---
title: "Verkehrssignalsteuerung neu denken: Vom festen Timing zur adaptiven Intelligenz"
description: "Eine Reflexion über die Entwicklung der Verkehrssignalsteuerung – von Schleifendetektoren und festen Plänen bis hin zu Reinforcement Learning und vernetzten autonomen Fahrzeugen."
pubDate: 2026-04-02
tags: ["Verkehrstechnik", "Verstärkungslernen", "Adaptive Steuerung", "Intelligente Stadt"]
category: Tech
---

# Verkehrssignalsteuerung neu denken: Vom festen Timing zur adaptiven Intelligenz

Verkehrssignale gibt es überall – wir begegnen ihnen Dutzende Male am Tag, meist ohne darüber nachzudenken. Aber wenn Sie jemals um 2 Uhr morgens an einer roten Ampel gesessen haben, ohne dass jemand in Sicht war, oder sich in einer „grünen Welle“ befanden, die perfekt von einer Kreuzung zur nächsten fließt, haben Sie bereits die Konsequenzen gespürt, die daraus resultieren, wie Verkehrsampeln optimiert werden (oder nicht).

Nachdem ich jahrelang mit Verkehrssimulationstools wie SUMO und CARLA gearbeitet und mich mit der Forschung zum Reinforcement Learning für die Signalsteuerung beschäftigt habe, bin ich zu dem Schluss gekommen, dass dieses Problem eine der interessantesten und am wenigsten erforschten Herausforderungen in der städtischen Mobilität ist. Hier ist meine ehrliche Überlegung darüber, wo wir stehen und wohin wir gehen könnten.

## Der traditionelle Ansatz: Festzeit- und Aktuatorsteuerung

Die meisten Verkehrssignale funktionieren auch heute noch nach einem von zwei Paradigmen:**Festzeitsteuerung** weist Grünphasen gemäß vorprogrammierten Zeitplänen zu, die typischerweise aus historischen Verkehrszählungen abgeleitet werden. Diese Zeitpläne werden oft einmal im Jahr aktualisiert – wenn überhaupt. Sie sind in dem Sinne robust, dass sie vorhersehbar und einfach zu bedienen sind, aber sie reagieren grundsätzlich auf die Vergangenheit, nicht auf die Gegenwart.

**Aktivierte Steuerung** fügt an Kreuzungen Schleifendetektoren oder Videokameras hinzu. Wenn ein Fahrzeug erkannt wird, verlängert das Signal die Grünphase. Es ist besser als eine feste Zeit, aber im Grunde ist es immer noch lokal – jede Kreuzung optimiert sich isoliert, ohne zu wissen, was stromaufwärts oder stromabwärts passiert.

Beide Ansätze haben eine grundlegende Einschränkung gemeinsam: **Sie optimieren für die Kreuzung, nicht für das Netzwerk.** Ein grünes Licht, das eine Kreuzung freigibt, könnte eine Warteschlange erzeugen, die zurückschwappt und drei andere blockiert. Verkehr ist ein System, keine Ansammlung unabhängiger Knoten.## Das netzwerkweite Problem: Warum Koordination alles verändert

Denken Sie darüber nach, was während einer typischen morgendlichen Hauptverkehrszeit passiert. Fahrzeuge strömen aus Wohngebieten auf Ausfallstraßen, und wenn diese Ausfallsignale nicht koordiniert werden, kommt es zu einem Phänomen, das als „progressiver Bandausfall“ bezeichnet wird – das genaue Gegenteil einer grünen Welle. Stop-and-go-Verkehr entsteht nicht aufgrund der hohen Nachfrage, sondern aufgrund einer schlechten Signalsteuerung.

Hier haben **SCOOT** (Split Cycle Offset Optimization Technique) und **SCATS** (Sydney Coordinated Adaptive Traffic System) ihre Spuren hinterlassen. Diese in den 1980er Jahren entwickelten Systeme nutzen Echtzeit-Erkennungsdaten, um Zykluslängen, Teilungen und Versätze in einem Netzwerk von Kreuzungen anzupassen. Sie sind wirklich effektiv – Städte, die SCOOT einsetzen, berichten von einer Reduzierung der Verspätungen um 10–20 %.Aber hier ist der Haken: SCOOT und SCATS basieren immer noch auf **Verkehrsflussmodellen** – makroskopischen oder mesoskopischen Näherungen der Art und Weise, wie sich Fahrzeuge bewegen. Diese Modelle wurden für den konventionellen Verkehr kalibriert. Sie kämpfen mit:

- **Übersättigungsbedingungen** (wenn die Nachfrage die Kapazität übersteigt)
- **Einmalige Staus** (Störungen, Bauarbeiten, Ereignisse)
- **Gemischter Verkehr** (von Menschen gesteuerte Fahrzeuge teilen sich die Fahrspur mit autonomen Fahrzeugen)
- **Langstreckenabhängigkeiten** (ein Engpass 3 Kreuzungen flussaufwärts)

Der modellbasierte Ansatz hat eine Grenze erreicht. Um weiter zu gehen, müssen wir die Komfortzone des Modells verlassen.

## Reinforcement Learning: Eine andere Art von OptimiererHier überschneiden sich meine eigenen Forschungserfahrungen und das Gesamtbild. Als ich an der SUMO-Python-Co-Simulationsplattform für die Messung städtischer Autobahnauffahrten arbeitete, begann ich mich zu fragen: Kann ein Agent allein aus Erfahrung lernen, Verkehrssignale zu steuern, ohne ein explizites Modell?

Die Idee hinter **Reinforcement Learning (RL)** für die Ampelsteuerung ist elegant:

- Der **Agent** ist der Ampelcontroller
- Der **Status** ist der aktuelle Verkehrszustand – Warteschlangenlängen, Wartezeiten, Fahrzeugpositionen, möglicherweise Fahrzeug-zu-Infrastruktur-Daten (V2I).
- Die **Aktion** ist die Signalphase, zu der gewechselt werden soll
- Die **Belohnung** ist eine Kombination von Metriken: Gesamtverzögerung minimieren, Durchsatz maximieren, Warteschlangenüberlauf bestrafenDer Agent muss die zugrunde liegende Dynamik des Verkehrsflusses nicht kennen. Es erlernt eine Kontrollpolitik direkt aus Interaktionen mit der Umgebung – genau wie AlphaGo gelernt hat, Go zu spielen, ohne dass ihm bei jedem Schritt gesagt wurde, was der „beste Zug“ war.

### Was es schwer macht

Es läuft nicht alles reibungslos. Die Verkehrssignalanlage RL steht vor mehreren praktischen Herausforderungen:

**Beispieleffizienz.** Im Gegensatz zu einem Spiel, bei dem Millionen von Episoden zum Selbstspielen möglich sind, erfordert der Einsatz in der realen Welt, dass der Agent zunächst in der Simulation lernt. Die Erstellung einer originalgetreuen Simulation ist nicht trivial – Spurwechselverhalten, Aggressivität des Fahrers, Unvorhersehbarkeit von Fußgängern, alles muss modelliert werden.

**Multi-Agenten-Koordination.** Ein einzelner Schnittpunkt ist eine Sache. Aber ein Netzwerk aus 50 Kreuzungen, von denen jede über einen eigenen RL-Agenten verfügt, führt zu einem RL-Problem mit mehreren Agenten. Die Agenten müssen koordinieren und nicht nur individuell optimieren. Die Aktion jedes Agenten beeinflusst die Beobachtungen seiner Nachbarn.**Sicherheit und Interpretierbarkeit.** Verkehrskontrolle ist sicherheitskritisch. Sie können einen Lernagenten nicht frei an einer echten Kreuzung experimentieren lassen. Die Basislinie muss sicher sein und das Lernen muss eingeschränkt werden – z. B. durch konservative Richtlinienaktualisierungen, Human-in-the-Loop-Fallback oder Sicherheitsschilde.

**Verallgemeinerung.** Ein RL-Agent, der auf Daten zur morgendlichen Hauptverkehrszeit geschult ist, kann mittags oder an einem Feiertagswochenende spektakulär ausfallen. Die Verschiebung der Verteilung ist ein echtes Problem.

### Vielversprechende Richtungen

Trotz der Herausforderungen bin ich wirklich gespannt, wohin die Reise führt. Ein paar Richtungen finde ich besonders vielversprechend:**Graphische neuronale Netze für räumliches Bewusstsein.** Anstatt jeder Kreuzung einen flachen Vektor ihrer eigenen Warteschlangenlängen zuzuführen, ermöglichen GNNs den Agenten, über die Netzwerktopologie zu kommunizieren und Informationen darüber auszutauschen, was an benachbarten Kreuzungen passiert. Auf diese Weise näherte ich mich während meines Praktikums bei Bosch China der Trajektorienerzeugung, und dieser Ansatz lässt sich natürlich auch auf die Signalsteuerung übertragen.

**Hybrides, physikinformiertes RL.** Durch die Kombination von First-Principles-Verkehrsmodellen (z. B. Store-and-Forward- oder Zellübertragungsmodellen) mit RL erhalten Sie das Beste aus beidem: Das Modell stellt Struktur- und Sicherheitsbeschränkungen bereit, während RL die feinkörnige Optimierung übernimmt. An dieser Stelle steht mein SCI-Artikel zum Thema Autobahnzufahrtsmessung – Q-Learning, unterstützt durch SUMO-Simulation, mit Kanalisierungsmodellierung.**V2I- und CAV-fähige Steuerung.** Mit der Marktdurchdringung vernetzter autonomer Fahrzeuge (CAVs) ändert sich die Rückkopplungsschleife dramatisch. Anstatt den Verkehrszustand anhand dünner Schleifendetektoren abzuleiten, können Signale Echtzeit-Positions- und Geschwindigkeitsdaten von jedem Fahrzeug im Netzwerk empfangen. Dabei handelt es sich nicht nur um eine schrittweise Verbesserung, sondern um eine grundlegende Veränderung dessen, was beobachtbar und kontrollierbar ist.

## Was wir aufgebaut haben und was bleibt

In meiner eigenen Arbeit – von der SUMO-CARLA-Fusionsplattform bis zum RL-basierten Ramp-Metering-Papier – habe ich sowohl das Potenzial als auch die Lücken aus erster Hand gesehen. Simulationsplattformen entwickeln sich schnell weiter. Mit der TraCI-Schnittstelle von SUMO können Sie alles in Python skripten. CARLA fügt die Sensortreue hinzu, die für eine wahrnehmungsbasierte Steuerung erforderlich ist. Die Tools sind nicht mehr der Flaschenhals.

Was aus meiner Sicht noch offen bleibt:1. **Benchmark-Umgebungen** – wir benötigen standardisierte Traffic-Netzwerk-Benchmarks mit konsistenten Metriken, wie etwa ML mit ImageNet und GLUE. Die Literatur ist voll von Spielzeugproblemen mit nur einer Kreuzung, die sich nicht auf den realen Einsatz übertragen lassen.

2. **Fairness und Gerechtigkeit** – die meisten RL-Signalcontroller optimieren die durchschnittliche Verzögerung. Aber ein Signal, das den vorherrschenden Verkehrsfluss bedient, könnte Fußgänger, Radfahrer oder Fahrzeuge auf Nebenstraßen systematisch bestrafen. Mehrobjektives RL mit Fairnessbeschränkungen ist noch wenig erforscht.

3. **Übertragung von der Simulation in die Realität.** Dies ist das Problem der letzten Meile. Eine Richtlinie, die in SUMO funktioniert, scheitert in der realen Welt oft aufgrund einer Lücke zwischen Simulation und Realität. Domänenrandomisierung, Systemidentifikation und robustes RL sind Teil der Lösung.4. **Öffentliche Akzeptanz.** Adaptive Signale, die ihr Verhalten nicht deterministisch ändern, können Fahrer verwirren. Neben der Kontrolltheorie muss es einen Forschungsstrang zu menschlichen Faktoren geben.

## Abschließende Gedanken

Die Steuerung von Verkehrsampeln ist eines dieser Probleme, die auf den ersten Blick einfach erscheinen, aber trügerisch tiefgreifend sind. Es handelt sich um ein Kontrollproblem, ein Netzwerkproblem, ein Fairnessproblem und zunehmend um ein Problem des maschinellen Lernens. Die Tatsache, dass an den meisten Kreuzungen der Welt noch immer Zeitmessmechanismen aus dem 19. Jahrhundert betrieben werden, ist ein Beweis sowohl für deren Zuverlässigkeit als auch für die Schwierigkeit, es noch besser zu machen.

Ich bin optimistisch. Die Konvergenz von billigen Sensoren, V2X-Kommunikation, besserer Simulation und intelligenteren RL-Algorithmen bietet eine echte Chance, die städtische Mobilität auf der grundlegendsten Ebene zu überdenken – grünes Licht nach dem anderen.

---*Wenn Sie an der Ampelsteuerung, RL für den Transport oder SUMO/CARLA-Simulation arbeiten, können Sie sich gerne an uns wenden. Immer gerne für den Austausch.*