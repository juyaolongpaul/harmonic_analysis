\version "2.10.33"

\header {
  title = "193 - Was bist du doch, o Seele, so betrübet"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c {
  \repeat volta 2 {
    r4 e''4 c4. b8 
    a4 e'4 f4 e8 d8 
    c4 d4 e4 e4 }
  r4 c4 d4. g,8 
  g2 r4 d'4   %5
  e4. c8 c2 
  r4 e4 f4 e4 
  d4 c4 b4. a8 
  gis8 e'8 d8 c8 b2 
  a2. 
}

alto = \relative c {
  \repeat volta 2 {
    r4 a''4 a4 gis4 
    e4 a4 a4 gis4 
    e4 a4 gis4 gis4 }
  r4 a4 g4 f4~ 
  f4 e4 r4 g4   %5
  g4 g4 f2 
  r4 g4 f8 a8 gis4 
  a8 d,8 e4 f4 e4 
  d4 a'2 gis4 
  e2. 
}

tenor = \relative c {
  \repeat volta 2 {
    r4 c'8 d8 e4 e8. d16 
    c4 c4 b4 b4 
    a8 c8 b8 a8 b4 b4 }
  r4 e4 d4 d4 
  c2 r4 d4   %5
  c4 c8 bes8 a2 
  r4 cis4 d4. c8~ 
  c8 b4 a4 gis4 a8 
  b4 e4 e4 e8 d8 
  cis2. 
}

baixo = \relative c {
  \repeat volta 2 {
    r4 a4 a'4 e4 
    a,4 a'4 d,4 e4 
    a4 f4 e4 e4 }
  r4 a,4 b2 
  c2 r4 b4   %5
  c4 e4 f2 
  r4 e4 d4 e4 
  f4 e4 d4 c4 
  b4 a4 e'4 e,4 
  a2. 
}

\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}