\version "2.10.33"

\header {
  title = "168 - Heut' ist, o Mensch, ein großer"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \minor
}


soprano = \relative c' {
  \partial 4 d4 
    f4 g4 a4 bes4 
  a4 g4 c4 bes4 
  a2 r4 d4 
  c4 bes4 a4 g4 
  f4 g4 f4 ees4   %5
  d2 r4 a'4 
  bes4 c4 d4 d4 
  c4 g8 a8 bes4 a4 
  g2.
  
}

alto = \relative c' {
  \partial 4 d4 
    a8 f'4 e8 f4 d8 e8 
  fis4 g4. a4 g8 
  fis2 r4 bes4~ 
  bes8 a4 g4 f4 ees8~ 
  ees8 d8 ees8 d8 c8 bes8 c8 a8   %5
  bes2 r4 c4 
  d8 f8 g8 a8 bes4 g8 f8 
  ees8 f8 g4 g4. fis8 
  d2.
  
}

tenor = \relative c' {
  \partial 4 g4 
    f8 a8 d8 c8 c4 s 
  a16 bes16 c4 bes16 a16 g8 d'8 d4 
  d2 r4 f4~ 
  f8. ees16 d4~ d8. c16 bes4~ 
  bes4 bes4 f2   %5
  f2 r4 f4 
  f'4 ees4 f4 d4 
  g,4 c4 d4. c8 
  b2.
  
}

baixo = \relative c {
  \partial 4 g4 
    d'4 bes8 c8 f4 g4 
  d4 e4. fis8 g4 
  d2 r4 bes4 
  f'4 g4 d4 ees4 
  bes8 a8 g4 a8 g8 a8 f8   %5
  bes2 r4 f'8 ees8 
  d4 c4 bes4 b4 
  c8 d8 ees4 bes8 c8 d4 
  g,2. 
  
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