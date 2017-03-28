\version "2.10.33"

\header {
  title = "68 - Wenn wir in höchsten Nöten sein"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key f \major
}


soprano = \relative c' {
  \partial 4 f4 
    f4 g4 a4 g8 a8 
  bes4 a4 g4. g8 
  f2 r4 a4 
  bes4 a4 g4 f4 
  e4 f4 g4 c4   %5
  bes4 a4 g8 f8 e4 
  f4 d4 c4 a'4 
  bes4 a4 g4 f4 
  a4 g4 f
  
}

alto = \relative c' {
  \partial 4 c4~
  c8 b8 c4. d8 e4 
  f8 g4 f4 e16 d16 e4 
  c2 r4 f4~ 
  f8 g~ g f~ f e8 d4 
  g,4 c4 c4 e8 fis8  %5
  g8 e8 f8 e8 d4~ d8 c8~
  c d16 c16 b4 g4 f'8 ees8 
  d8 e8 f4~ f8 e8~ e8 d16 e16 
  f4. e8 c4 
}

tenor = \relative c' {
  \partial 4 a8  g8 
    f4 c'8 bes8 a8 bes8 c4 
  f,8 c'8 c4 d8 bes8 g8 c16 bes16 
  a2 r4 c4 
  d8 c8 c4 d8 c4 b8 
  c4 f,4 e4 a4   %5
  d8 cis8 d4 g,4 g4 
  a4 g8 f8 e4 c'4 
  bes4 c4~ c8. bes16 a8 bes8 
  c4 c8. bes16 a4
  
}

baixo = \relative c {
  \partial 4 f8  e8 
    d4 e8 c8 f4 c4 
  d8 e8 f4 bes,4 c4 
  f,2 r4 f'8 ees8 
  d8 e8 f4 b,8 c8 d4 
  c8 bes8 a8 f8 c'4 a4   %5
  g4 d'8 c8 b4 c4 
  f,4 g4 c4 f,4 
  g4 a8 bes8 c4 d4 
  a8 bes8 c4 f,4
  
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