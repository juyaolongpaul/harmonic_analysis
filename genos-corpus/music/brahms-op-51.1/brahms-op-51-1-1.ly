\version "2.12.0"

\include "global.ly"
\include "violinoum.ly"
\include "violinodois.ly"
\include "viola.ly"
\include "violoncelo.ly"

%% comandos

%% para pizzicato use \pizz, para arco \arco
pizz = ^\markup{\italic{pizz.}}
arco = ^\markup{\italic{arco}}

%% score

\book {
  \header {
    title = "Quarteto n.1 em Dó menor"
    composer = "Johannes Brahms"
    opus = "Opus 51"
    copyright = "© Genos"
  }
  \score {
    \new StaffGroup <<
      \new Staff <<
        \set Staff.midiInstrument = "violin"
        \global
        \violinoum
      >>
      \new Staff <<
        \set Staff.midiInstrument = "violin"
        \global
        \violinodois
      >>
      \new Staff <<
        \set Staff.midiInstrument = "viola"
        \global
        \viola
      >>
      \new Staff <<
        \set Staff.midiInstrument = "cello"
        \global
        \violoncelo
      >>
    >>
    \midi { }
    \layout { }
  }
}
