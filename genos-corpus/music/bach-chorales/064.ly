\version "2.10.33"

\header {
  title = "64 - Freu' dich sehr, o meine Seele"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \major
}


soprano = \relative c'' {
  \repeat volta 2 {
    g4 a4 b4 a4 
    g4 fis4 e4 d4 
    g4 a4 b4 c4 
    b4 a4 g2 }
  b4 c4 d4 c4   %5
  b4 a4 b2 
  d4 d4 e4 d4 
  c4 b4 a2 
  b8 c8 d4 c4 b4 
  g4 a4 b2   %10
  g2 b4 d4 
  c4 b4 a4 b4 
  a2 g2 
  
}

alto = \relative c' {
  \repeat volta 2 {
    d4 d4 d4. cis8 
    b4 a8 d8 d8 cis8 d4 
    b4 d4 d4 d4 
    d4 d4 d2 }
  g4 g4 fis4 e8 fis8   %5
  g4 fis4 g2 
  g4 g4 g4 fis4 
  e4 d4 d2 
  d4 g8 f8 e4 d4 
  e8 dis8 e4. dis16 cis16 dis4   %10
  b2 e4 f4 
  e4 b8 c8 d4 d4 
  d2 d2 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    b4 fis4 g8 fis8 e4 
    d8 e8 fis8. g16 a8 e8 fis4 
    g4 fis4 g4 a4 
    a8 g8 g8 fis8 b2 }
  d4 e4 a,8 b8 c4   %5
  d4 d4 d2 
  b4 b4 c4 a4 
  g4 g4 fis2 
  g8 a8 b4. a4 g8~ 
  g8 fis8 e8 fis8 g8 fis16 e16 fis4   %10
  e2 g4 g4 
  g4 g4 fis4 g4~ 
  g8 fis16 e16 fis4 b2 
  
}

baixo = \relative c' {
  \repeat volta 2 {
    g4 d4 g,4 a4 
    b8 cis8 d4 a4 d,4 
    e'4 d8 c8 b8 a8 g8 fis8 
    g4 d4 g2 }
  g'8 fis8 e4 d4 a4   %5
  b8 c8 d4 g,2 
  g8 a8 b8 g8 c8 b8 c8 d8 
  e8 fis8 g4 d2 
  g4 g,4 a4 b4 
  c4 c4 b2   %10
  e2 e8 d8 c8 b8 
  c8 d8 e8 d8 c8 b8 a8 g8 
  d'4 d,4 g2 
  
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