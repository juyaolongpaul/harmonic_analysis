\version "2.10.33"

\header {
  title = "137 - Wer Gott vertraut, hat wohl gebaut"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c {
  \repeat volta 2 {
    r4 g''4 g4 a4 
    b2 r4 b4 
    c4 b4 a2 
    r4 a4 g4 a4 
    b4 a4 g2     %5
    a4 a4 b4 a4 
    g4 g4 fis2 
    g1 }
  r4 b4 c4 b4 
  a2 r4 a4   %10
  g4 g4 fis2 
  r4 fis4 g4 fis4 
  e4 e4 dis2 
  e2 r4 g4 
  g4 fis4 g2   %15
  r4 b4 c4 b4 
  a2 r4 a4 
  g4 a4 b4 a4 
  g2 a4 a4 
  b4 a4 g4 g4   %20
  fis2 g2 
  
}

alto = \relative c {
  \repeat volta 2 {
    r4 d'4 e8 g4 fis8 
    g2 r4 g4~ 
    g8 fis8 g4 g4 fis4 
    r4 fis4. e4 d8 
    d4 d4 d4 cis4     %5
    d4 d4 d8 e8 fis4~ 
    fis8 e8 d8 e8 d2 
    d1 }
  r4 g4 g8 a4 g8~ 
  g8 fis16 e16 fis4 r4 fis4~   %10
  fis4 e2 dis4 
  r4 fis4 b,8 e8 dis4 
  e4 b8 c8 b2 
  b2 r4 b8 c8 
  d8 e8 d4 d2   %15
  r4 g4 a4. g8 
  g4 fis4 r4 fis4 
  g4 d4 d8 e8 fis8 e8 
  d8 c8 d8 e8 fis4 a4~ 
  a8 g4 fis4 e8 d8 e8   %20
  d2 d2 
  
}

tenor = \relative c {
  \repeat volta 2 {
    r4 b'4 b4 d4 
    d2 r4 d4 
    c4 d4 d2 
    r4 d8 c8 b4 a4~ 
    a8 g4 fis8 g8 b8 a8 g8     %5
    fis4 d'8 c8 b8 cis8 d8 c8 
    b4 b8 a8 a8 b8 c4 
    b1 }
  r4 d4 e8 d8 d4 
  d2 r4 b4   %10
  b4 b4 b2 
  r4 b8 a8 g8 c4 b8~ 
  b8 a8 g8 fis8 fis8 g8 a4 
  g2 r4 g8 a8 
  b8 a8 a16 b16 c8 c4 b4   %15
  r4 d8 e4 d8 d4 
  d2 r4 d4 
  d4. c8 b8 c8 d4 
  g,8 a8 b8 c16 b16 a4 d4 
  d4 d4 b4. a8   %20
  a8 b8 c4 b2 
  
}

baixo = \relative c {
  \repeat volta 2 {
    r4 g'8 fis8 e4 d4 
    g2 r4 g,4 
    a4 b8 c8 d2 
    r4 d4 e4 fis4 
    g4 d4 e2     %5
    d4 fis4 g4 d4 
    e4 b8 c8 d2 
    g,1 }
  r4 g'8 fis8 e8 fis8 g4 
  d2 r4 dis8 b8   %10
  e8 fis8 g8 a8 b2 
  r4 dis,4 e4 b4 
  c4 g8 a8 b2 
  e2 r4 e4 
  b8 c8 d4 g,2   %15
  r4 g'4. fis8 g4 
  d2 r4 c4 
  b4 fis4 g4 d'8 c8 
  b8 a8 g4 d'4 fis4 
  g4 d4 e4 b8 c8   %20
  d2 g,2 
  
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