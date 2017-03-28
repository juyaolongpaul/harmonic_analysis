\version "2.10.33"

\header {
  title = "338 - Jesus, meine Zuversicht"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    a4 g8 fis8 b4 cis4 
    d4 d4 cis2 
    b8 cis8 d4 a8 g8 fis4 
    g4 fis4 e4 d4 }
  fis4 gis4 ais4 b8 cis8   %5
  d4 cis4 b2 
  cis4 d4 e4 fis4 
  e4 e4 d2 
  
}

alto = \relative c' {
  \repeat volta 2 {
    fis4 e8 d8 d8 e4 fis8 
    fis4. e8 e2 
    e4 d4 e4 d4 
    d4 d4 d8 cis8 a4 }
  b8 fis'4 e8 fis4 fis8 e8   %5
  fis4 fis4 fis2 
  g4 a4 a4 a4 
  b4 a8 g8 fis2 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    a4 a4. g4 a16 g16 
    fis8 b8 a8 gis8 a2 
    b4 a4 a4 a4 
    g4 a8 b8 b8 a16 g16 fis4 }
  b4 b4 cis4 b4   %5
  b4 ais8 e'8 d2 
  e4 fis4 e4 d4 
  d4 cis4 a2 
  
}

baixo = \relative c {
  \repeat volta 2 {
    d4 d4 g,4 a4 
    b4 e4 a,2 
    g'4 fis4 cis4 d8 c8 
    b8 bes8 a8 gis8 a4 d4 }
  dis4 e4. d8 g4   %5
  fis8 e8 fis4 b,2 
  e4 d4 d8 cis8 d4 
  g4 a4 d,2 
  
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