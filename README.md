Można zauważyć, pliki .lcvrp zawierają idealne ( albo najlepsze dotychczas znalezione ) permutacje. Dzięki temu, podzielenie permutacji na ciągłe fragmenty da nam optymalnie matematyczny podział do grupy. Możemy wykorzystać do tego algorytm Bellmana-Forda (O(n^2)), lub szybszy algorytm Split (O(n)) autorstwa Thibauta Vidala. Jednakże, w 3 wersji paczki konkursowej permutacja z pliku jest tasowana, przez co dostajemy nieuporządkowaną geometrycznie i raczej słabej jakości kolejność odwiedzin. Niemniej jednak uważam, że Split jest nadal potężnym narzędziem, jednak używanie go trzeba dostosować do specyfiki problemu.

Najważniejsze stosowane operatory:

1) Crossover:

\- geometryczny, bazujący na sąsiadach. Losuje losowego klienta od rodzica 1 a następnie przepisuje do dziecka id grup jego k najbliższych sasiadow. Reszta jest wtedy brana od rodzica 2.

\- SREX. Wymienia całe trasy. Około połowa tras jest brana od rodzica 1, następnie od rodzica 2 brane jest jak najwięcej tras, które nie kolidują z istniejącymi. Pozostali klienci, którzy kolidowali są dobierani do grup za pomocą heurystyki regret 3.

2) Mutacje:

\- MicroSplit. Realizuje wyżej opisany algorytm, ale na mniejszym oknie. Dzięki czemu możemy trafić na "dobry" kawałek permutacji i go optymalnie podzielić.

\- MergeSplit. Łączy 2 grupy i na ich klientach wykonuje optymalny podzial. Jest dobry również do "tasowania" i dzielenia ponownie, ponieważ rangi (kolejność w permutacji) mogą się zakleszczać.

\- RuinRecreate - standard w wariantach CVRP. Usuwa przydział do grup klienta i jego k najbliższych sąsiadów ( k zależy od intensywności) a następnie przydziela ich z powrotem z uwzględnieniem pojemności grup oraz ilości sąsiadów geometrycznych w innych grupach.

\- AdaptiveLNS - idea jest podobna do standardowego RR, ale moment niszczenia jest zależny od wybranej strategii. Możemy niszczyć m.in. klastry w permutacji, przepełnione grupy oraz najgorsze trasy pod względem dystansu i zapełnienia

\- ReturnMinimizer - usuwa powroty z ciężarówek np. wypełnienie 130% zmienia na 100%.

\- EliminateReturns - Zwiększenie upakowania (fill-rate). Usuwa końcówki tras, gdzie samochód wraca do bazy wioząc powietrze (np. wykorzystanie < 70%). Warto zaznaczyć, że z perspektywy evaluatora jedna trasa wypełniona w 200% da taki sam rezultat co 2 wozy upakowane w 100% ( pod warunkiem że kawałki permutacji się nie nałożą)

3) LocalSearch - prawdopodobnie najważniejszy element, wybiera stosowany operator w zależności od jego skuteczności (AOS) :

\- VND. Dla standardowych rozmiarów instancji przenosi klienta z 1 grupy do drugiej, lub zamienia 2 klientów grupami. Dla bezpiecznych ruchów, czyli takich które nie przekroczą pojemności, sotsuje delta evaluation, a dla niebezpiecznych symuluje koszt całej trasy.

\- Path Relinking sprawdza różnice między osobnikiem i guide solution (najlepszym na danej wyspie) i stara się połączyć te rozwiązania i znaleźć w konsekwencji lepsze.

\- Ejection Chains - jest to łańcuhcowa zamiana klientów

\- 3Swap, 4Swap. Wybiera geometrycznie bliskich klientów i testuje ich różne kombinacje (3! oraz 4! ruchów są akceptowalne wydajnościowo)

W projekcie postawiłem na heterogeniczny model wyspowy. Każda wyspa jest obslugiwana przez osobny wątek i ma osobną rolę / zadania. Wyspy o id parzystych to "Explorerzy" a nieparzystych nazywam "Exploatorami"

Explorator. Jak nazwa wskazuje, jego zadaniem jest szukanie nowych, obiecujących kawałków przestrzeni rozwiązań - duża szansa na mutacje, szybkie vnd z relocate + swap, większa populacja.

Exploit. Poleruje i odkrywa głębokie optima lokalne - mniejsza szansa na mutacje, ciężkie vnd z ejection chains, PR 3,4 Swap.

Zarządzanie różnorodnością - jest to bardzo ważny element aby uniknąć przedwczesnej zbieżności. Oparty jest na metryce złamanych pair ( Broken Pairs Distance ). W przypadku "wymarcia populacji" stosowana jest katastrofa. Po 5 nieudanych katastrofach wyspa przerzuca swoich osobników na inne wyspy, zaczyna kompletnie od nowa i blokuje migrację na 60 sekund.

Migracja - jest asynchroniczna w zależności od zdrowia konkretnej wyspy oprócz fitnessu, warunkiem wejścia na wyspę jest odpowiednia różnica w metryce BPD. Wprowadzone są również broadcasty - gdy któraś wyspa znajdzie znacząco lepsze rozwiązanie od poprzedniego globalnego current best, wysyła je do pozostałych wysp.

Inne, ciekawe pomysły:

RoutePool - klasyfikuje i zapisuje najlepsze dotychczasowe trasy. Idea została zaczerpnięta z MIP. Trasy oceniane są na podstawie załadowania tras jak najlepszego fitnessu. Następnie, co jakiś czas składa z nich "Frankensteina" (poprzez połączenie nienakładających się ze sobą tras) z nich za pomocą Beamsearcha. Wpuszczany jest do populacji jeśli jest wystarczająco dobry, ale również z 10% szansą siłowo wstrzykuje go do populacji. Dzięki temu, dostarcza dobre cechy do rozwiązań, co w procesach localsearch i mieszania np. crossoverem SREX może dać swietne wyniki. Analizując działanie programu, wielokrotnie widoczne było że po długim czasie stagnacji "Frankenstein" był wstrzykiwany do populacji, a od razu potem na tej wyspie znaleziono globalny best. Wprowadzona jest również migracja genów między wyspami, ponieważ w teorii każda wyspa przeszukuje inną przestrzeń rozwiązań.

Deduplikacja:

Mechanizmy cachowania (na 64 bitach) wprowadzone są w 3 miejscach - przy tworzeniu dzieci, przy evaluowaniu całych rozwiązań oraz do cachowania pojedynczych tras w moim Evaluatorze. Wykorzystywanie różnych sposobów hashowania zapewnia większe bezpieczeństwo kolizji hashy, a hashowanie tras jest kluczowe dla działania VND, gdzie symulowane jest cała trasa w przypadku niebezpiecznych ruchów.

Sposoby inicjalizajci:

Dla zachowania różnorodności, wybierane metody inicjalizacji osobników są różne - np. Random, Chunked czy K-Center (nie wykorzystuje centroidow jak K-means dzięki czemu działa dla instancji explicit). Inny ciekawy mechanizm stosowany to tworzenie tymczasowej permutacji za pomocą double bridge a następnie poddawanie jej algorytmowi split. Dzięki temu, przynajmniej w teorii, trasy mają jakiś matematyczny sens.
