\version "2.10.33"

\header {
  title = "33 - Herr, ich habe mißgehandelt"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    a4 e8 fis8 gis4 a4 
    b8 c16 d16 c8 b8 b4 a4 
    c4 b4 a4 b4 
    c4 d4 e2 }
  e4 f4 g4 c,4   %5
  f4 e4 d4 c4 
  d4 d4 e4 a,4 
  d4 c8 b8 b2 
  a2. r4
}

alto = \relative c' {
  \repeat volta 2 {
    e8 d8 c4 b4 a4 
    e'4 e8 f8 e4 e4 
    a8 g8 f8 e8 e8 a4 gis8 
    a8 e8 a8 b16 a16 gis2 }
  a4 a4 g4 a8 g8   %5
  a8 d8 g,4. g16 f16 e4 
  g8 a8 b4. gis8 a8 g8 
  f8 e8 e8 f8 e4. d8 
  cis2. r4
}

tenor = \relative c' {
  \repeat volta 2 {
    c8 b8 a4 e8 e'8 d8 c8 
    b8 gis8 a4. gis8 c4 
    e4 d4 c4 f,8 e8 
    e8 a4 gis16 a16 b2 }
  c4 d4. c8 c4   %5
  c8 b8 c4. b8 g4 
  b8 d8 g8 f8 e4. d8 
  c8 b8 a4 a4 gis4 
  e2. r4
}

baixo = \relative c {
  \repeat volta 2 {
    a8 b8 c8 d8 e4 fis4 
    gis8 e8 a8 d,8 e8 e,8 a4 
    a8 a'4 gis8 a8 f8 d8 e8 
    a,8 g8 f4 e2 }
  a8 a'8 g8 f8 e4 f8 e8   %5
  d4 e8 f8 g8 g,8 c4 
  g'8 f8 e8 d8 c8 d8 c8 b8 
  a8 gis8 a8 d8 e4 e,4 
  a2. r4
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