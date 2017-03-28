\version "2.10.33"

\header {
  title = "134 - Du geballtes Weltgebäude"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \minor
}


soprano = \relative c'' {
  \repeat volta 2 {
    d4 a4 d4 d4 
    e4 c4 b2 
    a2 f4 a4 
    g4 f4 e4. d8 
    d1 }  %5
  f4 g4 a4 a4 
  bes4 a4 g2 
  f2 f4 g4 
  a4 a4 bes4 a4 
  g2 f2   %10
  a4 b4 c4 a4 
  d4 d4 cis2 
  d8 e8 f4 e4 d4 
  d4 cis4 d2 
  
}

alto = \relative c' {
  \repeat volta 2 {
    f4 a4 a4 gis8 a8 
    b8 e,8 e4 f4 e8 d8 
    cis2 d4 d8 cis8 
    d8 e4 d8 d4 cis4 
    a1 }  %5
  d4 e4 f4 f4 
  f8 g4 f8 f4 e4 
  c2 d4. c8~ 
  c8 f8 a4. g4 f8~ 
  f8 e16 d16 e4 c2   %10
  f4 f4 g4. f8 
  f4 f8 e8 e2 
  f8 g8 a4 g4 f4 
  e4 e4 fis2 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    a4 f'8 e8 d8 c8 b4~ 
    b4. a8 a4 gis4 
    e2 a4 a4 
    d,8 a'8 a4 bes4 a8. g16 
    f1 }  %5
  a4 c4 c4 c4 
  d8 c8 c4 d8 bes8 g8 c16 bes16 
  a2 a4 g4 
  f8 a8 d8 dis8 d4 c4 
  d4 c8 bes8 a2   %10
  c4 d4 c4 c4 
  d8 c8 b4 a2 
  a4 d8 c8 bes4 a4 
  bes4 a4 a2 
  
}

baixo = \relative c {
  \repeat volta 2 {
    d4. c8 b4. a8 
    gis4 a4 d4 e4 
    a,2 d8 e8 f4 
    b,8 cis8 d4 g,4 a4 
    d1 }  %5
  d4 c8 bes8 a8 g8 f8 e8 
  d8 e8 f4 bes4 c4 
  f,2 d'4 e4 
  f4 fis4 g4 a4 
  bes4 c4 f,2   %10
  f8 e8 d4 e8 c8 f4 
  bes8 a8 gis4 a2 
  f8 e8 d4. cis8 d8 f8 
  g8 e8 a4 d,2 
  
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