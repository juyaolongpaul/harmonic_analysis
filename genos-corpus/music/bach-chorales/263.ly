\version "2.10.33"

\header {
  title = "263 - Jesu, meine Freude"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key e \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    b4 b4 a4 g4 
    fis2 e2 
    b'4 cis4 d4 b4 
    e2 dis2 
    e8 fis8 g4 fis4. fis8     %5
    e1 }
  b4 b4 c4 b4 
  a4. a8 g2 
  b4 cis4 d4 b4 
  e4 d8 cis8 cis2   %10
  b2 b4 b4 
  a4 g8 fis8 fis2 
  e1 
  
}

alto = \relative c'' {
  \repeat volta 2 {
    g4 fis4 e8 dis8 e4 
    e4 dis4 b2 
    g'8 fis8 e4 d4 d4 
    g8 a8 b4 b2 
    g8 a8 b4 b4. a8     %5
    g1 }
  g4 g4 a4 g4 
  g4 fis4 d2 
  g4 g4 a4 g8 a8 
  b4 b4 b4 ais4   %10
  fis2 g4 fis4 
  e4 e4 e4 dis4 
  b1 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    e4 b4 c8 fis,8 g4 
    c4 b8 a8 g2 
    e'8 d8 cis8 b8 a4 g8 a8 
    b4 g'4 fis2 
    e4 e4 e4 dis4     %5
    b1 }
  e4 d4 d4 d4 
  e4 d8 c8 b2 
  d4 e4 d4 d4 
  g4 fis4 g4 fis8 e8   %10
  dis2 e4 fis8 g8 
  a8 a,8 b4 c4 b8 a8 
  gis1 
  
}

baixo = \relative c {
  \repeat volta 2 {
    e4 d4 c4. b8 
    a4 b4 e2 
    e4 a8 g8 fis4 g8 fis8 
    e8 fis8 g8 a8 b2 
    c4 b8 a8 b4 b,4     %5
    e1 }
  e8 fis8 g4 fis4 g4 
  c,4 d4 g,2 
  g'4 fis8 e8 fis4 g8 fis8 
  e4 b4 e4 fis4   %10
  b,2 e4 d4 
  c4 b4 a4 b4 
  e1 
  
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