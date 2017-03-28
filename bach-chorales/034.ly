\version "2.10.33"

\header {
  title = "34 - Erbarm' dich mein, o Herre Gott"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 e4 
    g4 g4 a4 b4 
    c4 b4 a4 g4 
    c4 b4 c4 a4 
    g4 f4 e4 } e4 
  a4 a4 g4 c4   %5
  b4 a4 g4 g4 
  c4 g4 a4 e4 
  a8 g8 f4 e2 
  a4 a4 g8 f8 e4 
  f4 d4 c4 c4   %10
  g'2 a4 b4 
  c4 g4 a4 g4 
  f4. f8 e4
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 b4 
    e4 e4 e4 f8 e8 
    e8 a4 gis8 e4 e8 f8 
    g8 a a g16 f16 g4. f8~ 
    f8 e4 d16 c16 b4 } c4 
  f4 f4 f8 e16 d16 e8 fis8   %5
  g16 d16 g4 fis8 d4 e4 
  e4 e8 d8 c8 b8 cis8 d8 
  e4. d8~ d cis8 c4 
  c8 ees4 d8 d4 c4 
  c4. b8 g4 g4   %10
  d'4 e8 d8 c4 d4 
  e8 f8 g8 e8 f4. e8~ 
  e8 d8~ d8 c8 b4
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 g4 
    b4 b4 c4 d4 
    c8 e8 f8 e16 d16 c4 b4 
    e4 f8 e16 d16 c4 c4 
    c8. bes16 a8 b16 a16 gis4 } a4 
  c4 d4. c8 g8 a8   %5
  d,8 d'8 e8 d16 c16 b4 c4 
  c4 c8 bes8 a4 a4 
  a4 a4 a4 a4 
  a8 c4 b16 a16 b4 c8 b8 
  a4 g8. f16 e4 e4   %10
  g4 c8 d8 e8 f8 g4 
  g,4 c4 c4 c8. bes16 
  a8 b16 c16 b8 a8 gis4
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 e4~
    e8 f8 e8 d8 c8 a'8~ 
    a8 gis8 a8 c,8 d8 e8 a,4 
    e'8 d8 c4 d8 g8 e8 c8 
    f4 c4 d4 e4 }
  a8 g8 f8 e8 d8 c8 b8 c8   %5
  b8 a8 g8 b8 c8 d8 g,4 
  c8 b8 a8 b8 c8 e,8 f8 g8 
  a8 b8 cis8 a8 d4 a4 
  a'8 g8 f4 fis4 g4 
  a8 g8 f8 d8 g8 g,8 c4   %10
  c4 b4 c8 b8 a4 
  g8 f8 e8 d8 e8 c8 f8 a8 
  c4 d2 e4
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